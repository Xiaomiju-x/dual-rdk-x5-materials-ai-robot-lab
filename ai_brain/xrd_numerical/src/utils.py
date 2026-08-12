"""
工具函数: 可视化、配置加载、IO
"""

import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from .extract_peaks import PeakInfo


def load_config(config_path: str = "config.yaml") -> dict:
    """加载YAML配置文件"""
    with open(config_path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def plot_raw_spectrum(two_theta: np.ndarray, intensity: np.ndarray,
                      title: str = "XRD Spectrum", save_path: str = None):
    """绘制原始XRD谱图"""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(two_theta, intensity, 'b-', linewidth=0.5)
    ax.set_xlabel('2θ (°)')
    ax.set_ylabel('Intensity')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"图片已保存: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_peaks(two_theta: np.ndarray, intensity_processed: np.ndarray,
               peaks: list[PeakInfo], title: str = "Peak Detection",
               save_path: str = None):
    """绘制处理后的谱图和检测到的峰"""
    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(two_theta, intensity_processed, 'b-', linewidth=0.5, label='Processed')

    if peaks:
        peak_pos = [p.position for p in peaks]
        peak_int = [p.intensity for p in peaks]
        ax.scatter(peak_pos, peak_int, c='red', s=30, zorder=5, label=f'Peaks ({len(peaks)})')

        # 标注前10个最强峰的2θ值
        for p in peaks[:10]:
            ax.annotate(f'{p.position:.1f}°', xy=(p.position, p.intensity),
                       xytext=(0, 10), textcoords='offset points',
                       fontsize=7, ha='center', color='red')

    ax.set_xlabel('2θ (°)')
    ax.set_ylabel('Normalized Intensity')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"图片已保存: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str],
                          save_path: str = None):
    """绘制混淆矩阵"""
    fig, ax = plt.subplots(figsize=(max(8, len(class_names)), max(6, len(class_names) * 0.8)))

    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           xlabel='Predicted',
           ylabel='True',
           title='Confusion Matrix')

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    # 在格子中标注数字
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha='center', va='center',
                   color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"图片已保存: {save_path}")
    else:
        plt.show()
    plt.close()
