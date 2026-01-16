"""
Training Report Generator

Automatically generates a PDF report summarizing training runs,
including model configuration, training parameters, and loss curves.
"""

import os
import io
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for PDF generation
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class TrainingMetrics:
    """Container for training metrics at a specific step"""
    step: int
    loss: float
    learning_rate: float
    perplexity: Optional[float] = None
    accuracy: Optional[float] = None
    tokens_seen: Optional[int] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class EvalMetrics:
    """Container for evaluation metrics at a specific step"""
    step: int
    loss: float
    perplexity: Optional[float] = None
    accuracy: Optional[float] = None
    timestamp: float = field(default_factory=time.time)


class TrainingReport:
    """
    Collects training metrics and generates a PDF report.

    Usage:
        report = TrainingReport(
            training_type="base",
            model_config=model_config,
            training_config=training_config,
            output_dir="/app/data"
        )

        # During training loop:
        report.log_step(step, loss, lr, perplexity)
        report.log_eval(step, eval_loss, eval_perplexity)

        # At end of training:
        report.finalize(final_metrics, checkpoint_path)
        report.generate_pdf()
    """

    def __init__(
        self,
        training_type: str,  # "base", "sft", "ppo", "dpo", "grpo"
        model_config: Any,
        training_config: Any,
        output_dir: str = "/app/data",
    ):
        self.training_type = training_type
        self.model_config = model_config
        self.training_config = training_config
        self.output_dir = output_dir

        self.start_time = time.time()
        self.end_time: Optional[float] = None

        # Metrics storage
        self.train_metrics: List[TrainingMetrics] = []
        self.eval_metrics: List[EvalMetrics] = []

        # Final results
        self.final_metrics: Dict[str, Any] = {}
        self.checkpoint_path: Optional[str] = None

        # Hardware info
        self.gpu_info = self._get_gpu_info()

    def _get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information"""
        try:
            import torch
            if torch.cuda.is_available():
                return {
                    "name": torch.cuda.get_device_name(0),
                    "memory_total": f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB",
                    "cuda_version": torch.version.cuda,
                }
        except:
            pass
        return {"name": "Unknown", "memory_total": "N/A", "cuda_version": "N/A"}

    def log_step(
        self,
        step: int,
        loss: float,
        learning_rate: float,
        perplexity: Optional[float] = None,
        accuracy: Optional[float] = None,
        tokens_seen: Optional[int] = None,
    ):
        """Log metrics for a training step"""
        self.train_metrics.append(TrainingMetrics(
            step=step,
            loss=loss,
            learning_rate=learning_rate,
            perplexity=perplexity,
            accuracy=accuracy,
            tokens_seen=tokens_seen,
        ))

    def log_eval(
        self,
        step: int,
        loss: float,
        perplexity: Optional[float] = None,
        accuracy: Optional[float] = None,
    ):
        """Log metrics for an evaluation"""
        self.eval_metrics.append(EvalMetrics(
            step=step,
            loss=loss,
            perplexity=perplexity,
            accuracy=accuracy,
        ))

    def finalize(
        self,
        final_metrics: Optional[Dict[str, Any]] = None,
        checkpoint_path: Optional[str] = None,
    ):
        """Finalize the report with final metrics"""
        self.end_time = time.time()
        self.final_metrics = final_metrics or {}
        self.checkpoint_path = checkpoint_path

    def _get_config_value(self, config: Any, key: str, default: Any = "N/A") -> Any:
        """Safely get a config value from dataclass or dict"""
        if config is None:
            return default
        if hasattr(config, key):
            return getattr(config, key)
        if isinstance(config, dict):
            return config.get(key, default)
        return default

    def _format_duration(self, seconds: float) -> str:
        """Format duration in human readable format"""
        td = timedelta(seconds=int(seconds))
        hours, remainder = divmod(td.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if td.days > 0:
            return f"{td.days}d {hours}h {minutes}m"
        elif hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"

    def _format_number(self, n: float) -> str:
        """Format large numbers with K/M/B suffixes"""
        if n >= 1e9:
            return f"{n/1e9:.2f}B"
        elif n >= 1e6:
            return f"{n/1e6:.2f}M"
        elif n >= 1e3:
            return f"{n/1e3:.2f}K"
        return f"{n:.2f}"

    def _create_loss_plot(self) -> Optional[bytes]:
        """Create loss curve plot and return as bytes"""
        if not self.train_metrics:
            return None

        fig, ax = plt.subplots(figsize=(8, 4))

        # Training loss
        steps = [m.step for m in self.train_metrics]
        losses = [m.loss for m in self.train_metrics]
        ax.plot(steps, losses, 'b-', alpha=0.7, label='Training Loss', linewidth=1)

        # Eval loss
        if self.eval_metrics:
            eval_steps = [m.step for m in self.eval_metrics]
            eval_losses = [m.loss for m in self.eval_metrics]
            ax.plot(eval_steps, eval_losses, 'r-', marker='o', label='Eval Loss', linewidth=2, markersize=4)

        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title('Training & Evaluation Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Save to bytes
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()

    def _create_perplexity_plot(self) -> Optional[bytes]:
        """Create perplexity curve plot and return as bytes"""
        train_ppl = [(m.step, m.perplexity) for m in self.train_metrics if m.perplexity is not None]
        eval_ppl = [(m.step, m.perplexity) for m in self.eval_metrics if m.perplexity is not None]

        if not train_ppl and not eval_ppl:
            return None

        fig, ax = plt.subplots(figsize=(8, 4))

        if train_ppl:
            steps, ppls = zip(*train_ppl)
            ax.plot(steps, ppls, 'b-', alpha=0.7, label='Training Perplexity', linewidth=1)

        if eval_ppl:
            steps, ppls = zip(*eval_ppl)
            ax.plot(steps, ppls, 'r-', marker='o', label='Eval Perplexity', linewidth=2, markersize=4)

        ax.set_xlabel('Step')
        ax.set_ylabel('Perplexity')
        ax.set_title('Training & Evaluation Perplexity')
        ax.legend()
        ax.grid(True, alpha=0.3)

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()

    def _create_lr_plot(self) -> Optional[bytes]:
        """Create learning rate schedule plot"""
        if not self.train_metrics:
            return None

        steps = [m.step for m in self.train_metrics]
        lrs = [m.learning_rate for m in self.train_metrics]

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(steps, lrs, 'g-', linewidth=1.5)
        ax.set_xlabel('Step')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()

    def generate_pdf(self, filename: Optional[str] = None) -> str:
        """
        Generate PDF report and save to output directory.

        Returns:
            Path to the generated PDF file
        """
        try:
            from fpdf import FPDF
        except ImportError:
            print("Warning: fpdf2 not installed. Skipping PDF report generation.")
            print("Install with: pip install fpdf2")
            return ""

        # Generate filename
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_report_{self.training_type}_{timestamp}.pdf"

        output_path = os.path.join(self.output_dir, filename)

        # Create PDF
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        # Title
        pdf.set_font('Helvetica', 'B', 20)
        pdf.cell(0, 15, 'LLM-Lab Training Report', ln=True, align='C')

        pdf.set_font('Helvetica', '', 12)
        pdf.cell(0, 8, f'Training Type: {self.training_type.upper()}', ln=True, align='C')
        pdf.cell(0, 8, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', ln=True, align='C')
        pdf.ln(10)

        # Model Configuration Section
        pdf.set_font('Helvetica', 'B', 14)
        pdf.set_fill_color(240, 240, 240)
        pdf.cell(0, 10, 'Model Configuration', ln=True, fill=True)
        pdf.set_font('Helvetica', '', 10)

        model_info = [
            ("Architecture", self._get_config_value(self.model_config, 'model_architecture', 'transformer')),
            ("Attention Type", self._get_config_value(self.model_config, 'attention_type', 'gqa')),
            ("Hidden Size (d_model)", self._get_config_value(self.model_config, 'd_model')),
            ("Layers", self._get_config_value(self.model_config, 'n_layers')),
            ("Attention Heads", self._get_config_value(self.model_config, 'n_heads')),
            ("KV Heads", self._get_config_value(self.model_config, 'n_kv_heads')),
            ("FFN Size (d_ff)", self._get_config_value(self.model_config, 'd_ff')),
            ("Vocab Size", self._get_config_value(self.model_config, 'vocab_size')),
            ("Max Sequence Length", self._get_config_value(self.model_config, 'max_seq_len')),
            ("Positional Encoding", self._get_config_value(self.model_config, 'positional_encoding')),
            ("Activation", self._get_config_value(self.model_config, 'activation')),
            ("Normalization", self._get_config_value(self.model_config, 'norm_type')),
            ("MoE Enabled", self._get_config_value(self.model_config, 'use_moe', False)),
        ]

        # Add MoE details if enabled
        if self._get_config_value(self.model_config, 'use_moe', False):
            model_info.extend([
                ("Num Experts", self._get_config_value(self.model_config, 'num_experts')),
                ("Experts per Token", self._get_config_value(self.model_config, 'num_experts_per_token')),
            ])

        for label, value in model_info:
            pdf.cell(80, 7, f"  {label}:", ln=False)
            pdf.cell(0, 7, str(value), ln=True)

        pdf.ln(5)

        # Training Configuration Section
        pdf.set_font('Helvetica', 'B', 14)
        pdf.cell(0, 10, 'Training Configuration', ln=True, fill=True)
        pdf.set_font('Helvetica', '', 10)

        training_info = [
            ("Optimizer", self._get_config_value(self.training_config, 'optimizer',
                          self._get_config_value(self.training_config, 'policy_optimizer'))),
            ("Learning Rate", self._get_config_value(self.training_config, 'lr',
                              self._get_config_value(self.training_config, 'learning_rate'))),
            ("Batch Size", self._get_config_value(self.training_config, 'batch_size')),
            ("Gradient Accumulation", self._get_config_value(self.training_config, 'gradient_accumulation_steps')),
            ("Max Steps", self._get_config_value(self.training_config, 'max_steps')),
            ("Warmup Steps", self._get_config_value(self.training_config, 'warmup_steps')),
            ("Scheduler", self._get_config_value(self.training_config, 'scheduler')),
            ("Weight Decay", self._get_config_value(self.training_config, 'weight_decay')),
            ("Gradient Clipping", self._get_config_value(self.training_config, 'grad_clip',
                                   self._get_config_value(self.training_config, 'max_grad_norm'))),
        ]

        # Add LoRA info if applicable
        use_lora = self._get_config_value(self.training_config, 'use_lora', False)
        if use_lora:
            training_info.extend([
                ("LoRA Enabled", "Yes"),
                ("LoRA Rank", self._get_config_value(self.training_config, 'lora_rank')),
                ("LoRA Alpha", self._get_config_value(self.training_config, 'lora_alpha')),
            ])

        for label, value in training_info:
            if value is not None and value != "N/A":
                pdf.cell(80, 7, f"  {label}:", ln=False)
                pdf.cell(0, 7, str(value), ln=True)

        # Dataset info
        datasets = self._get_config_value(self.training_config, 'datasets', [])
        if datasets:
            pdf.ln(3)
            pdf.set_font('Helvetica', 'B', 10)
            pdf.cell(0, 7, "  Datasets:", ln=True)
            pdf.set_font('Helvetica', '', 10)
            for ds in datasets[:5]:  # Limit to 5
                if isinstance(ds, dict):
                    ds_name = ds.get('name', str(ds))
                else:
                    ds_name = str(ds)
                pdf.cell(0, 6, f"    - {ds_name}", ln=True)

        pdf.ln(5)

        # Training Results Section
        pdf.set_font('Helvetica', 'B', 14)
        pdf.cell(0, 10, 'Training Results', ln=True, fill=True)
        pdf.set_font('Helvetica', '', 10)

        # Duration
        if self.end_time:
            duration = self.end_time - self.start_time
            pdf.cell(80, 7, "  Training Duration:", ln=False)
            pdf.cell(0, 7, self._format_duration(duration), ln=True)

        # Final step
        if self.train_metrics:
            pdf.cell(80, 7, "  Final Step:", ln=False)
            pdf.cell(0, 7, str(self.train_metrics[-1].step), ln=True)

            pdf.cell(80, 7, "  Final Training Loss:", ln=False)
            pdf.cell(0, 7, f"{self.train_metrics[-1].loss:.4f}", ln=True)

        # Final eval metrics
        if self.eval_metrics:
            last_eval = self.eval_metrics[-1]
            pdf.cell(80, 7, "  Final Eval Loss:", ln=False)
            pdf.cell(0, 7, f"{last_eval.loss:.4f}", ln=True)

            if last_eval.perplexity:
                pdf.cell(80, 7, "  Final Eval Perplexity:", ln=False)
                pdf.cell(0, 7, f"{last_eval.perplexity:.2f}", ln=True)

            if last_eval.accuracy:
                pdf.cell(80, 7, "  Final Eval Accuracy:", ln=False)
                pdf.cell(0, 7, f"{last_eval.accuracy:.4f}", ln=True)

        # Tokens seen
        if self.train_metrics and self.train_metrics[-1].tokens_seen:
            pdf.cell(80, 7, "  Total Tokens Seen:", ln=False)
            pdf.cell(0, 7, self._format_number(self.train_metrics[-1].tokens_seen), ln=True)

        # Additional final metrics
        for key, value in self.final_metrics.items():
            if key not in ['val_loss', 'val_perplexity', 'val_accuracy']:
                pdf.cell(80, 7, f"  {key}:", ln=False)
                if isinstance(value, float):
                    pdf.cell(0, 7, f"{value:.4f}", ln=True)
                else:
                    pdf.cell(0, 7, str(value), ln=True)

        pdf.ln(5)

        # Hardware Section
        pdf.set_font('Helvetica', 'B', 14)
        pdf.cell(0, 10, 'Hardware', ln=True, fill=True)
        pdf.set_font('Helvetica', '', 10)

        pdf.cell(80, 7, "  GPU:", ln=False)
        pdf.cell(0, 7, self.gpu_info.get('name', 'Unknown'), ln=True)
        pdf.cell(80, 7, "  GPU Memory:", ln=False)
        pdf.cell(0, 7, self.gpu_info.get('memory_total', 'N/A'), ln=True)
        pdf.cell(80, 7, "  CUDA Version:", ln=False)
        pdf.cell(0, 7, self.gpu_info.get('cuda_version', 'N/A'), ln=True)

        # Checkpoint path
        if self.checkpoint_path:
            pdf.ln(3)
            pdf.cell(80, 7, "  Checkpoint:", ln=False)
            pdf.set_font('Helvetica', '', 8)
            pdf.cell(0, 7, self.checkpoint_path, ln=True)
            pdf.set_font('Helvetica', '', 10)

        # Add graphs on new page
        pdf.add_page()
        pdf.set_font('Helvetica', 'B', 14)
        pdf.cell(0, 10, 'Training Curves', ln=True, fill=True)
        pdf.ln(5)

        # Loss plot
        loss_plot = self._create_loss_plot()
        if loss_plot:
            loss_path = os.path.join(self.output_dir, '_temp_loss_plot.png')
            with open(loss_path, 'wb') as f:
                f.write(loss_plot)
            pdf.image(loss_path, x=10, w=190)
            os.remove(loss_path)
            pdf.ln(5)

        # Perplexity plot
        ppl_plot = self._create_perplexity_plot()
        if ppl_plot:
            ppl_path = os.path.join(self.output_dir, '_temp_ppl_plot.png')
            with open(ppl_path, 'wb') as f:
                f.write(ppl_plot)
            pdf.image(ppl_path, x=10, w=190)
            os.remove(ppl_path)
            pdf.ln(5)

        # Learning rate plot
        lr_plot = self._create_lr_plot()
        if lr_plot:
            lr_path = os.path.join(self.output_dir, '_temp_lr_plot.png')
            with open(lr_path, 'wb') as f:
                f.write(lr_plot)
            pdf.image(lr_path, x=10, w=190)
            os.remove(lr_path)

        # Footer
        pdf.set_y(-25)
        pdf.set_font('Helvetica', 'I', 8)
        pdf.cell(0, 10, 'Generated by LLM-Lab - https://github.com/your-repo/llm-lab', ln=True, align='C')

        # Save PDF
        pdf.output(output_path)
        print(f"\n📄 Training report saved to: {output_path}")

        return output_path


def create_report(
    training_type: str,
    model_config: Any,
    training_config: Any,
    output_dir: str = "/app/data",
) -> TrainingReport:
    """
    Factory function to create a TrainingReport instance.

    Args:
        training_type: Type of training ("base", "sft", "ppo", "dpo", "grpo")
        model_config: Model configuration
        training_config: Training configuration
        output_dir: Directory to save the report

    Returns:
        TrainingReport instance
    """
    return TrainingReport(
        training_type=training_type,
        model_config=model_config,
        training_config=training_config,
        output_dir=output_dir,
    )
