"""
=============================================================================
肺结节 CT 影像良恶性分析 Streamlit 网站
=============================================================================
"""

# ============================================
# 第一步：导入所有需要的库
# ============================================

import os
import io
import pickle
import urllib.request
import tempfile
from pathlib import Path

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

# 影像处理
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.measure import regionprops, moments, moments_central, moments_hu
from skimage.filters import sobel
from scipy import stats
from scipy.stats import skew, kurtosis
from scipy.ndimage import gaussian_filter

# SHAP
import shap

# ============================================
# 第二步：设置页面
# ============================================

st.set_page_config(
    page_title="肺结节良恶性分析系统",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# 第三步：CSS样式
# ============================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .benign {
        background-color: #d4edda;
        border: 2px solid #28a745;
        color: #155724;
    }
    .malignant {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        color: #721c24;
    }
    .info-box {
        background-color: #e7f3ff;
        border: 1px solid #b8daff;
        border-radius: 5px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# 第四步：模型下载配置（GitHub Releases）
# ============================================

# GitHub Releases 下载链接（使用GitHub用户名和仓库名）
# 格式：https://github.com/用户名/仓库名/releases/download/标签名/文件名
GITHUB_USERNAME = "CaLek-bot"  # 修改为你的GitHub用户名
REPO_NAME = "lung-nodule-analysis"  # 修改为你的仓库名
RELEASE_TAG = "v1.0"  # 发布标签
MODEL_FILENAME = "best_model_SVM.pkl"  # 模型文件名

MODEL_URL = f"https://github.com/{GITHUB_USERNAME}/{REPO_NAME}/releases/download/{RELEASE_TAG}/{MODEL_FILENAME}"

# 本地模型路径（相对路径，适配云端环境）
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)


def download_model():
    """从GitHub Releases下载模型（首次运行时）"""
    # 检查本地是否已有模型
    if os.path.exists(MODEL_PATH):
        st.success("模型文件已加载！")
        return

    # 创建模型目录
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 尝试从GitHub下载
    try:
        with st.spinner("首次运行，正在从GitHub下载模型（约 10-50MB）..."):
            # 添加请求头，模拟浏览器访问
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            req = urllib.request.Request(MODEL_URL, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as response:
                with open(MODEL_PATH, 'wb') as f:
                    f.write(response.read())
        st.success("模型下载完成！")
    except Exception as e:
        st.error(f"模型下载失败: {str(e)}")
        st.info("""
        **解决方法：**
        1. 手动下载模型文件
        2. 上传到GitHub Releases（标签: v1.0）
        3. 或修改MODEL_PATH为本地绝对路径
        """)
        # 提供手动上传选项
        uploaded_model = st.file_uploader("或手动上传模型文件(.pkl)", type=['pkl'])
        if uploaded_model is not None:
            with open(MODEL_PATH, 'wb') as f:
                f.write(uploaded_model.getvalue())
            st.success("模型上传成功！")
        else:
            st.stop()


# ============================================
# 第五步：调用下载函数
# ============================================

download_model()


# ============================================
# 第六步：影像组学特征提取器
# ============================================

class RadiomicsFeatureExtractor:
    """影像组学特征提取器"""

    def extract_all_features(self, image, mask=None):
        """提取所有特征"""
        features = {}

        # 如果没有提供掩膜，自动生成
        if mask is None:
            mask = self._auto_segment(image)

        mask = (mask > 0).astype(np.uint8)
        roi = image * mask
        roi_values = image[mask > 0]

        if len(roi_values) == 0:
            return {}

        features.update(self._first_order_features(roi_values))
        features.update(self._shape_features(mask))
        features.update(self._glcm_features(image, mask))
        features.update(self._glrlm_features(image, mask))
        features.update(self._lbp_features(image, mask))
        features.update(self._gradient_features(image, mask))
        features.update(self._wavelet_like_features(image, mask))

        return features

    def _auto_segment(self, image):
        """自动分割肺结节"""
        from skimage.filters import threshold_otsu
        try:
            thresh = threshold_otsu(image)
        except:
            thresh = np.median(image)

        mask = (image > thresh).astype(np.uint8)

        # 保留最大连通区域
        from skimage.measure import label
        labeled = label(mask)
        if labeled.max() > 0:
            regions = regionprops(labeled)
            largest = max(regions, key=lambda r: r.area)
            mask = (labeled == largest.label).astype(np.uint8)

        return mask

    def _first_order_features(self, values):
        f = {}
        prefix = 'firstorder_'
        f[prefix + 'Energy'] = np.sum(values ** 2)
        f[prefix + 'TotalEnergy'] = np.sum(values ** 2) * len(values)
        f[prefix + 'Entropy'] = stats.entropy(np.histogram(values, bins=32)[0] + 1e-10)
        f[prefix + 'Minimum'] = np.min(values)
        f[prefix + '10Percentile'] = np.percentile(values, 10)
        f[prefix + '90Percentile'] = np.percentile(values, 90)
        f[prefix + 'Maximum'] = np.max(values)
        f[prefix + 'Mean'] = np.mean(values)
        f[prefix + 'Median'] = np.median(values)
        f[prefix + 'InterquartileRange'] = np.percentile(values, 75) - np.percentile(values, 25)
        f[prefix + 'Range'] = np.max(values) - np.min(values)
        f[prefix + 'MeanAbsoluteDeviation'] = np.mean(np.abs(values - np.mean(values)))
        f[prefix + 'RobustMeanAbsoluteDeviation'] = np.mean(np.abs(values - np.median(values)))
        f[prefix + 'RootMeanSquared'] = np.sqrt(np.mean(values ** 2))
        f[prefix + 'StandardDeviation'] = np.std(values)
        f[prefix + 'Skewness'] = skew(values)
        f[prefix + 'Kurtosis'] = kurtosis(values)
        f[prefix + 'Variance'] = np.var(values)
        f[prefix + 'Uniformity'] = np.sum((np.histogram(values, bins=32)[0] / len(values)) ** 2)
        return f

    def _shape_features(self, mask):
        f = {}
        prefix = 'shape2D_'
        regions = regionprops(mask)
        if len(regions) == 0:
            return {prefix + 'Area': 0, prefix + 'Perimeter': 0}
        props = regions[0]
        f[prefix + 'Area'] = props.area
        f[prefix + 'Perimeter'] = props.perimeter if props.perimeter else 0
        f[prefix + 'MajorAxisLength'] = props.major_axis_length
        f[prefix + 'MinorAxisLength'] = props.minor_axis_length
        f[prefix + 'Eccentricity'] = props.eccentricity
        f[prefix + 'Orientation'] = props.orientation
        f[prefix + 'ConvexArea'] = props.convex_area
        f[prefix + 'Solidity'] = props.solidity
        f[prefix + 'EquivalentDiameter'] = props.equivalent_diameter
        f[prefix + 'Extent'] = props.extent
        if props.perimeter and props.perimeter > 0:
            f[prefix + 'Circularity'] = 4 * np.pi * props.area / (props.perimeter ** 2)
        else:
            f[prefix + 'Circularity'] = 0
        if props.minor_axis_length and props.minor_axis_length > 0:
            f[prefix + 'AspectRatio'] = props.major_axis_length / props.minor_axis_length
        else:
            f[prefix + 'AspectRatio'] = 0
        if props.bbox_area and props.bbox_area > 0:
            f[prefix + 'Rectangularity'] = props.area / props.bbox_area
        else:
            f[prefix + 'Rectangularity'] = 0
        try:
            m = moments(mask)
            cr = m[0, 1] / m[0, 0] if m[0, 0] > 0 else 0
            cc = m[1, 0] / m[0, 0] if m[0, 0] > 0 else 0
            mu = moments_central(mask, cr, cc)
            nu = moments_hu(mu)
            for i, hu in enumerate(nu):
                f[prefix + f'HuMoment{i + 1}'] = hu
        except:
            for i in range(7):
                f[prefix + f'HuMoment{i + 1}'] = 0
        return f

    def _glcm_features(self, image, mask):
        f = {}
        prefix = 'glcm_'
        roi = image.copy()
        roi[mask == 0] = 0
        roi_min, roi_max = roi[mask > 0].min(), roi[mask > 0].max()
        if roi_max > roi_min:
            roi_quantized = ((roi - roi_min) / (roi_max - roi_min) * 31).astype(np.uint8)
        else:
            roi_quantized = np.zeros_like(roi, dtype=np.uint8)
        roi_quantized[mask == 0] = 0
        distances = [1, 2, 3]
        angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
        glcm = graycomatrix(roi_quantized, distances=distances, angles=angles,
                            levels=32, symmetric=True, normed=True)
        props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
        for prop in props:
            values = graycoprops(glcm, prop)
            f[prefix + prop + '_Mean'] = np.mean(values)
            f[prefix + prop + '_Std'] = np.std(values)
            f[prefix + prop + '_Max'] = np.max(values)
            f[prefix + prop + '_Min'] = np.min(values)
        glcm_entropy = -np.sum(glcm * np.log(glcm + 1e-10))
        f[prefix + 'Entropy'] = glcm_entropy
        idm = np.sum(glcm / (1 + np.arange(32)[:, None, None, None] ** 2 +
                             np.arange(32)[None, :, None, None] ** 2))
        f[prefix + 'IDM'] = idm
        return f

    def _glrlm_features(self, image, mask):
        f = {}
        prefix = 'glrlm_'
        roi_values = image[mask > 0]
        if len(roi_values) == 0:
            return {prefix + 'SRE': 0, prefix + 'LRE': 0, prefix + 'GLN': 0,
                    prefix + 'RLN': 0, prefix + 'RP': 0, prefix + 'LGLRE': 0,
                    prefix + 'HGLRE': 0}
        roi_min, roi_max = roi_values.min(), roi_values.max()
        if roi_max > roi_min:
            quantized = ((roi_values - roi_min) / (roi_max - roi_min) * 15).astype(int)
        else:
            quantized = np.zeros_like(roi_values, dtype=int)
        unique, counts = np.unique(quantized, return_counts=True)
        total_runs = len(unique)
        total_pixels = len(quantized)
        if total_runs == 0:
            return {prefix + 'SRE': 0, prefix + 'LRE': 0, prefix + 'GLN': 0,
                    prefix + 'RLN': 0, prefix + 'RP': 0, prefix + 'LGLRE': 0,
                    prefix + 'HGLRE': 0}
        run_lengths = counts
        gray_levels = unique
        f[prefix + 'SRE'] = np.sum(1.0 / run_lengths ** 2) / total_runs
        f[prefix + 'LRE'] = np.sum(run_lengths ** 2) / total_runs
        f[prefix + 'GLN'] = np.sum(counts ** 2) / total_runs ** 2
        f[prefix + 'RLN'] = total_runs / total_pixels
        f[prefix + 'RP'] = total_runs / total_pixels
        f[prefix + 'LGLRE'] = np.sum(gray_levels ** 2 / run_lengths) / total_runs
        f[prefix + 'HGLRE'] = np.sum(gray_levels ** 2 * run_lengths) / total_runs
        return f

    def _lbp_features(self, image, mask):
        f = {}
        prefix = 'lbp_'
        roi = image.copy()
        roi_min, roi_max = roi.min(), roi.max()
        if roi_max > roi_min:
            roi_norm = ((roi - roi_min) / (roi_max - roi_min) * 255).astype(np.uint8)
        else:
            roi_norm = np.zeros_like(roi, dtype=np.uint8)
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(roi_norm, n_points, radius, method='uniform')
        lbp_roi = lbp[mask > 0]
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp_roi, bins=n_bins, range=(0, n_bins))
        hist = hist.astype(float)
        hist /= (hist.sum() + 1e-10)
        f[prefix + 'Mean'] = np.mean(lbp_roi)
        f[prefix + 'Std'] = np.std(lbp_roi)
        f[prefix + 'Entropy'] = stats.entropy(hist + 1e-10)
        f[prefix + 'Energy'] = np.sum(hist ** 2)
        for p in [10, 25, 50, 75, 90]:
            f[prefix + f'Percentile{p}'] = np.percentile(lbp_roi, p)
        return f

    def _gradient_features(self, image, mask):
        f = {}
        prefix = 'gradient_'
        gradient = sobel(image)
        gradient_roi = gradient[mask > 0]
        if len(gradient_roi) == 0:
            return {prefix + 'Mean': 0, prefix + 'Std': 0, prefix + 'Max': 0,
                    prefix + 'Min': 0, prefix + 'Energy': 0}
        f[prefix + 'Mean'] = np.mean(gradient_roi)
        f[prefix + 'Std'] = np.std(gradient_roi)
        f[prefix + 'Max'] = np.max(gradient_roi)
        f[prefix + 'Min'] = np.min(gradient_roi)
        f[prefix + 'Energy'] = np.sum(gradient_roi ** 2)
        f[prefix + 'Entropy'] = stats.entropy(np.histogram(gradient_roi, bins=32)[0] + 1e-10)
        return f

    def _wavelet_like_features(self, image, mask):
        f = {}
        prefix = 'wavelet_'
        roi = image.copy()
        roi[mask == 0] = 0
        sigmas = [1.0, 2.0, 4.0]
        for sigma in sigmas:
            filtered = gaussian_filter(roi, sigma=sigma)
            filtered_roi = filtered[mask > 0]
            if len(filtered_roi) > 0:
                f[prefix + f'LL_sigma{sigma}_Mean'] = np.mean(filtered_roi)
                f[prefix + f'LL_sigma{sigma}_Std'] = np.std(filtered_roi)
                f[prefix + f'LL_sigma{sigma}_Energy'] = np.sum(filtered_roi ** 2)
                f[prefix + f'LL_sigma{sigma}_Entropy'] = stats.entropy(
                    np.histogram(filtered_roi, bins=32)[0] + 1e-10)
        return f


# ============================================
# 第七步：为 pickle 兼容性添加的类定义
# ============================================

class FeatureSelector:
    """四级特征筛选器（最小化定义）"""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.selected_features = None
        self.feature_names = None
        self.lasso_model = None
        self.lasso_scaler = None
        self.rf_model = None


# ============================================
# 第八步：加载模型
# ============================================

@st.cache_resource
def load_model(model_path):
    """加载训练好的模型"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    return data


# ============================================
# 第九步：图像处理
# ============================================

def load_dicom(file):
    """加载 DICOM 文件"""
    dcm = pydicom.dcmread(file)
    img = dcm.pixel_array.astype(float)

    # 应用 VOI LUT
    if 'WindowCenter' in dcm and 'WindowWidth' in dcm:
        wc = dcm.WindowCenter
        ww = dcm.WindowWidth
        if isinstance(wc, pydicom.multival.MultiValue):
            wc = wc[0]
        if isinstance(ww, pydicom.multival.MultiValue):
            ww = ww[0]
        img_min = wc - ww // 2
        img_max = wc + ww // 2
        img = np.clip(img, img_min, img_max)

    # 归一化到 0-255
    img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
    return img


def preprocess_image(image):
    """预处理图像为模型输入格式"""
    # 转换为灰度图
    if len(image.shape) == 3:
        image = np.mean(image, axis=2)

    # 调整大小为 512x512
    from skimage.transform import resize
    image = resize(image, (512, 512), mode='constant', anti_aliasing=True)

    return image


# ============================================
# 第十步：SHAP 可视化
# ============================================

def plot_shap_waterfall(model, X, feature_names, scaler):
    """绘制 SHAP 瀑布图"""
    X_scaled = scaler.transform(X.reshape(1, -1))

    model_type = type(model).__name__

    if model_type in ['LGBMClassifier', 'XGBClassifier', 'RandomForestClassifier']:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_scaled)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
    else:
        # SVM/LR 用 KernelExplainer
        background = shap.sample(X_scaled, min(10, len(X_scaled)), random_state=42)
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(X_scaled, nsamples=100)

        if isinstance(shap_values, list):
            try:
                shap_values = np.array(shap_values[1])
            except:
                shap_values = np.array(shap_values[0])
        elif isinstance(shap_values, np.ndarray):
            shap_values = np.array(shap_values)

    # 处理各种形状情况
    if shap_values.ndim == 3:
        shap_values = shap_values[0, :, 1] if shap_values.shape[2] > 1 else shap_values[0, :, 0]
    elif shap_values.ndim == 2:
        if shap_values.shape[0] == 1:
            shap_values = shap_values[0]
        elif shap_values.shape[1] == 2:
            shap_values = shap_values[:, 1]
        else:
            shap_values = shap_values[0]

    # 确保是一维
    shap_values = np.atleast_1d(shap_values).flatten()

    # 确保 base_values 是标量
    if hasattr(explainer, 'expected_value'):
        ev = explainer.expected_value
        if isinstance(ev, (list, np.ndarray)):
            ev = np.atleast_1d(ev)
            base_val = float(ev[1]) if len(ev) > 1 else float(ev[0])
        else:
            base_val = float(ev)
    else:
        base_val = 0.0

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_values,
            base_values=base_val,
            data=X_scaled[0],
            feature_names=feature_names
        ),
        max_display=15,
        show=False
    )
    plt.title('SHAP 特征贡献瀑布图', fontsize=14)
    plt.tight_layout()
    return fig


# ============================================
# 第十一步：主界面
# ============================================

def main():
    # 标题
    st.markdown('<div class="main-header">肺结节 CT 影像良恶性分析系统</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">基于影像组学和机器学习的智能辅助诊断系统</div>', unsafe_allow_html=True)

    # 侧边栏
    with st.sidebar:
        st.header("系统设置")

        # 模型选择（使用配置的路径）
        model_path = st.sidebar.text_input(
            "模型文件路径",
            value=MODEL_PATH
        )

        st.markdown("---")
        st.header("使用说明")
        st.markdown("""
        1. 上传 CT 影像文件（DICOM 或图片格式）
        2. 系统自动提取影像组学特征
        3. AI 模型进行良恶性预测
        4. 查看预测结果和可解释性分析
        """)

        st.markdown("---")
        st.header("免责声明")
        st.info("本系统仅供科研和辅助诊断参考，不能替代专业医生的诊断意见。")

    # 主界面
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("上传影像")

        uploaded_file = st.file_uploader(
            "选择 CT 影像文件",
            type=['dcm', 'dicom', 'png', 'jpg', 'jpeg', 'tiff', 'bmp'],
            help="支持 DICOM 格式和常见图片格式"
        )

        if uploaded_file is not None:
            # 显示原始图像
            file_type = uploaded_file.name.lower().split('.')[-1]

            if file_type in ['dcm', 'dicom']:
                # DICOM 文件
                with tempfile.NamedTemporaryFile(delete=False, suffix='.dcm') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name

                image = load_dicom(tmp_path)
                os.unlink(tmp_path)
            else:
                # 普通图片
                image = np.array(Image.open(uploaded_file).convert('L'))

            # 预处理
            processed_image = preprocess_image(image)

            # 显示图像
            st.subheader("原始影像")
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(image, cmap='gray')
            ax.axis('off')
            ax.set_title('上传的 CT 影像')
            st.pyplot(fig)

            # 显示预处理后的图像
            st.subheader("预处理后的影像")
            fig2, ax2 = plt.subplots(figsize=(6, 6))
            ax2.imshow(processed_image, cmap='gray')
            ax2.axis('off')
            ax2.set_title('512x512 标准化影像')
            st.pyplot(fig2)

    with col2:
        st.header("分析结果")

        if uploaded_file is not None:
            with st.spinner('正在分析中，请稍候...'):
                try:
                    # 加载模型
                    model_data = load_model(model_path)
                    model = model_data['model']
                    scaler = model_data['scaler']
                    feature_names = model_data['feature_names']
                    model_type = type(model).__name__

                    # 提取特征
                    extractor = RadiomicsFeatureExtractor()
                    features = extractor.extract_all_features(processed_image)

                    if not features:
                        st.error("特征提取失败，请检查影像质量")
                        return

                    # 构建特征向量
                    X = np.array([features.get(name, 0) for name in feature_names]).reshape(1, -1)
                    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

                    # 预测
                    X_scaled = scaler.transform(X)
                    prob = model.predict_proba(X_scaled)[0, 1]
                    pred = model.predict(X_scaled)[0]

                    # 显示结果
                    if pred == 1:
                        result_class = "malignant"
                        result_text = "恶性 (Malignant)"
                        result_desc = "模型判断该结节为恶性，建议尽快进行进一步检查（如穿刺活检、PET-CT 等）。"
                    else:
                        result_class = "benign"
                        result_text = "良性 (Benign)"
                        result_desc = "模型判断该结节为良性，建议定期随访观察。"

                    st.markdown(f"""
                    <div class="result-box {result_class}">
                        <h2 style="margin:0;">{result_text}</h2>
                        <p style="margin:0.5rem 0 0 0; font-size:1.1rem;">{result_desc}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    # 概率显示
                    st.subheader("恶性概率")

                    prob_percent = prob * 100
                    col_prob1, col_prob2, col_prob3 = st.columns(3)

                    with col_prob1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{prob_percent:.1f}%</div>
                            <div class="metric-label">恶性概率</div>
                        </div>
                        """, unsafe_allow_html=True)

                    with col_prob2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{(1 - prob) * 100:.1f}%</div>
                            <div class="metric-label">良性概率</div>
                        </div>
                        """, unsafe_allow_html=True)

                    with col_prob3:
                        confidence = max(prob, 1 - prob) * 100
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{confidence:.1f}%</div>
                            <div class="metric-label">置信度</div>
                        </div>
                        """, unsafe_allow_html=True)

                    # 概率条
                    st.progress(float(prob))

                    # 风险分级
                    st.subheader("风险分级")
                    if prob < 0.2:
                        risk_level = "低风险"
                        risk_color = "green"
                        risk_advice = "建议 6-12 个月随访"
                    elif prob < 0.5:
                        risk_level = "中低风险"
                        risk_color = "orange"
                        risk_advice = "建议 3-6 个月随访"
                    elif prob < 0.8:
                        risk_level = "中高风险"
                        risk_color = "orange"
                        risk_advice = "建议 1-3 个月随访，考虑进一步检查"
                    else:
                        risk_level = "高风险"
                        risk_color = "red"
                        risk_advice = "建议立即进行进一步检查（穿刺活检/手术）"

                    st.markdown(f"""
                    <div class="info-box">
                        <h4 style="color:{risk_color}; margin:0;">{risk_level}</h4>
                        <p style="margin:0.5rem 0 0 0;">{risk_advice}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    # SHAP 可解释性
                    st.subheader("可解释性分析")

                    tab1, tab2 = st.tabs(["瀑布图", "特征重要性"])

                    with tab1:
                        fig_waterfall = plot_shap_waterfall(model, X[0], feature_names, scaler)
                        if fig_waterfall:
                            st.pyplot(fig_waterfall)
                            st.caption("瀑布图显示各特征对预测结果的贡献程度")

                    with tab2:
                        st.info("特征重要性排序功能")

                    # 特征详情
                    with st.expander("查看提取的特征详情"):
                        feature_df = pd.DataFrame({
                            '特征名': feature_names,
                            '特征值': X[0]
                        })
                        st.dataframe(feature_df, use_container_width=True)

                except Exception as e:
                    st.error(f"分析过程中出现错误: {str(e)}")
                    st.exception(e)
        else:
            st.info("请在左侧上传 CT 影像文件开始分析")

    # 底部信息
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; color:#888; font-size:0.9rem;">
        肺结节良恶性分析系统 | 基于 LIDC-IDRI 数据集训练 | 仅供科研参考
    </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()