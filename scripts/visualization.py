#!/usr/bin/env python3
"""
Script tạo biểu đồ và visualization cho kết quả đánh giá model
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import json

# Cấu hình matplotlib cho tiếng Việt
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

class ResultVisualizer:
    def __init__(self, results_file: str = "evaluation_results.csv", 
                 avg_metrics_file: str = "average_metrics.csv"):
        """
        Khởi tạo visualizer
        
        Args:
            results_file: File kết quả chi tiết
            avg_metrics_file: File thống kê trung bình
        """
        self.results_file = results_file
        self.avg_metrics_file = avg_metrics_file
        self.output_dir = Path("evaluation_plots")
        self.output_dir.mkdir(exist_ok=True)
        
    def load_data(self):
        """Load dữ liệu từ file"""
        try:
            self.df_results = pd.read_csv(self.results_file)
            self.df_avg = pd.read_csv(self.avg_metrics_file)
            print(f"Đã load {len(self.df_results)} kết quả chi tiết và {len(self.df_avg)} thống kê trung bình")
        except FileNotFoundError as e:
            print(f"Không tìm thấy file: {e}")
            return False
        return True
    
    def create_metric_comparison_plot(self):
        """Tạo biểu đồ so sánh các metric giữa các model"""
        if not hasattr(self, 'df_avg'):
            print("Chưa load dữ liệu")
            return
        
        metrics = ['bleu', 'rouge1', 'rougeL', 'cosine']
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('So sánh các Metric giữa các Model', fontsize=16, fontweight='bold')
        
        for i, metric in enumerate(metrics):
            ax = axes[i//2, i%2]
            
            # Tạo bar plot
            bars = ax.bar(self.df_avg['model'], self.df_avg[metric], 
                         color=sns.color_palette("husl", len(self.df_avg)))
            
            # Thêm giá trị trên bar
            for bar, value in zip(bars, self.df_avg[metric]):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            ax.set_title(f'{metric.upper()} Score', fontweight='bold')
            ax.set_ylabel('Score')
            ax.set_ylim(0, max(self.df_avg[metric]) * 1.2)
            ax.tick_params(axis='x', rotation=45)
            
            # Thêm grid
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'metric_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_overall_ranking_plot(self):
        """Tạo biểu đồ xếp hạng tổng hợp"""
        if not hasattr(self, 'df_avg'):
            print("Chưa load dữ liệu")
            return
        
        # Sắp xếp theo overall score
        df_sorted = self.df_avg.sort_values('overall_score', ascending=True)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Tạo horizontal bar plot
        bars = ax.barh(df_sorted['model'], df_sorted['overall_score'],
                      color=sns.color_palette("viridis", len(df_sorted)))
        
        # Thêm giá trị trên bar
        for i, (bar, value) in enumerate(zip(bars, df_sorted['overall_score'])):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{value:.4f}', ha='left', va='center', fontweight='bold')
        
        ax.set_title('Xếp hạng Model theo Điểm Tổng hợp', fontsize=16, fontweight='bold')
        ax.set_xlabel('Overall Score')
        ax.set_xlim(0, max(df_sorted['overall_score']) * 1.1)
        
        # Thêm grid
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'overall_ranking.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_radar_chart(self):
        """Tạo biểu đồ radar để so sánh đa chiều"""
        if not hasattr(self, 'df_avg'):
            print("Chưa load dữ liệu")
            return
        
        # Chuẩn hóa dữ liệu về thang điểm 0-1
        metrics = ['bleu', 'rouge1', 'rougeL', 'cosine']
        df_normalized = self.df_avg.copy()
        
        for metric in metrics:
            max_val = self.df_avg[metric].max()
            min_val = self.df_avg[metric].min()
            if max_val > min_val:
                df_normalized[metric] = (self.df_avg[metric] - min_val) / (max_val - min_val)
            else:
                df_normalized[metric] = 0.5
        
        # Tạo radar chart
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Góc cho các metric
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Đóng polygon
        
        # Vẽ cho từng model
        colors = sns.color_palette("husl", len(self.df_avg))
        
        for i, (_, row) in enumerate(self.df_avg.iterrows()):
            values = [df_normalized.loc[df_normalized['model'] == row['model'], metric].iloc[0] 
                     for metric in metrics]
            values += values[:1]  # Đóng polygon
            
            ax.plot(angles, values, 'o-', linewidth=2, label=row['model'], color=colors[i])
            ax.fill(angles, values, alpha=0.25, color=colors[i])
        
        # Cấu hình trục
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.upper() for m in metrics])
        ax.set_ylim(0, 1)
        ax.set_title('Biểu đồ Radar - So sánh đa chiều các Model', 
                    fontsize=16, fontweight='bold', pad=20)
        
        # Thêm legend
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'radar_chart.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_box_plots(self):
        """Tạo box plot để xem phân phối của các metric"""
        if not hasattr(self, 'df_results'):
            print("Chưa load dữ liệu")
            return
        
        metrics = ['bleu', 'rouge1', 'rougeL', 'cosine']
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Phân phối các Metric theo Model', fontsize=16, fontweight='bold')
        
        for i, metric in enumerate(metrics):
            ax = axes[i//2, i%2]
            
            # Tạo box plot
            sns.boxplot(data=self.df_results, x='model', y=metric, ax=ax)
            
            ax.set_title(f'Phân phối {metric.upper()}', fontweight='bold')
            ax.set_ylabel('Score')
            ax.tick_params(axis='x', rotation=45)
            
            # Thêm điểm trung bình
            means = self.df_results.groupby('model')[metric].mean()
            for j, (model, mean_val) in enumerate(means.items()):
                ax.text(j, mean_val, f'{mean_val:.3f}', ha='center', va='bottom', 
                       fontweight='bold', color='red')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'box_plots.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_correlation_heatmap(self):
        """Tạo heatmap tương quan giữa các metric"""
        if not hasattr(self, 'df_results'):
            print("Chưa load dữ liệu")
            return
        
        metrics = ['bleu', 'rouge1', 'rougeL', 'cosine']
        
        # Tính correlation matrix
        corr_matrix = self.df_results[metrics].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Tạo heatmap
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        
        ax.set_title('Ma trận Tương quan giữa các Metric', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_model_performance_summary(self):
        """Tạo bảng tóm tắt hiệu suất model"""
        if not hasattr(self, 'df_avg'):
            print("Chưa load dữ liệu")
            return
        
        # Tạo bảng tóm tắt
        summary_data = []
        metrics = ['bleu', 'rouge1', 'rougeL', 'cosine']
        
        for _, row in self.df_avg.iterrows():
            model_summary = {
                'Model': row['model'],
                'Overall Score': f"{row['overall_score']:.4f}",
                'Rank': f"{row['overall_rank']:.0f}"
            }
            
            # Thêm từng metric
            for metric in metrics:
                model_summary[f'{metric.upper()}'] = f"{row[metric]:.4f}"
                model_summary[f'{metric.upper()}_Rank'] = f"{row[f'{metric}_rank']:.0f}"
            
            summary_data.append(model_summary)
        
        # Tạo DataFrame và lưu
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_csv(self.output_dir / 'model_performance_summary.csv', index=False)
        
        # In ra console
        print("\n" + "="*100)
        print("BẢNG TÓM TẮT HIỆU SUẤT MODEL")
        print("="*100)
        print(df_summary.to_string(index=False))
        
        return df_summary
    
    def create_all_visualizations(self):
        """Tạo tất cả các biểu đồ"""
        if not self.load_data():
            return
        
        print("Bắt đầu tạo các biểu đồ...")
        
        # Tạo các biểu đồ
        self.create_metric_comparison_plot()
        self.create_overall_ranking_plot()
        self.create_radar_chart()
        self.create_box_plots()
        self.create_correlation_heatmap()
        self.create_model_performance_summary()
        
        print(f"\nHoàn thành! Tất cả biểu đồ được lưu trong thư mục: {self.output_dir}")

def main():
    """Hàm main"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Tạo biểu đồ cho kết quả đánh giá model")
    parser.add_argument("--results", type=str, default="evaluation_results.csv",
                       help="File kết quả chi tiết")
    parser.add_argument("--avg_metrics", type=str, default="average_metrics.csv",
                       help="File thống kê trung bình")
    
    args = parser.parse_args()
    
    visualizer = ResultVisualizer(args.results, args.avg_metrics)
    visualizer.create_all_visualizations()

if __name__ == "__main__":
    main() 