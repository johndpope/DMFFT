# Claude AI Assistant Guidelines for HRR Project

## Error Checking Protocol

### Before Declaring Success

**IMPORTANT**: Always verify that training and experiments are actually successful before reporting completion. Follow these steps:

1. **Check WandB Logs**:
   - Always examine WandB console output for errors
   - Look for Python exceptions (AttributeError, ValueError, etc.)
   - Check for warnings that might indicate issues
   - Verify that metrics are being logged correctly

2. **Common Error Patterns to Watch For**:
   - `AttributeError`: Often indicates model architecture mismatches
   - `'Sequential' object has no attribute 'weight'`: Module type mismatch
   - `RuntimeError`: GPU/memory issues or tensor shape mismatches
   - `ValueError`: Data format or configuration issues

3. **Verification Steps**:
   ```bash
   # Check training logs
   tail -n 100 training.log

   # Check for Python errors
   grep -i "error\|exception\|traceback" training.log

   # Monitor GPU usage
   nvidia-smi
   ```

4. **Model Architecture Compatibility**:
   - Enhanced models may use Sequential modules instead of single Linear layers
   - Visualization code must handle both types
   - Always test compatibility when switching between model variants

### Example Error Case

**Issue**: NMF visualization failing with enhanced model
```
AttributeError: 'Sequential' object has no attribute 'weight'
```

**Root Cause**: Enhanced model uses Sequential classification head, but NMF visualization assumed Linear layer

**Fix Applied**: Modified `visualize_linear_layer_nmf` to handle both:
```python
if isinstance(layer, nn.Sequential):
    linear_layers = [m for m in layer.modules() if isinstance(m, nn.Linear)]
    if linear_layers:
        layer = linear_layers[0]  # Use first Linear layer
```

## Training Monitoring Best Practices

1. **Real-time Monitoring**:
   - Keep checking background processes with `BashOutput` tool
   - Monitor loss convergence and validation metrics
   - Watch for gradient explosion/vanishing

2. **Checkpoint Validation**:
   - Verify checkpoint files are created
   - Test loading checkpoints before assuming training completed
   - Check model can perform inference after training

3. **Checkpoint Saving - IMPORTANT**:
   - **Only save the best checkpoint** - do not save every epoch
   - Avoid clogging HDD with multiple checkpoint files
   - Use `save_best_only=True` pattern in training loops
   - Delete old checkpoints after saving better ones

4. **Metric Validation**:
   - Ensure metrics make sense (e.g., accuracy between 0-100%)
   - Check for NaN or Inf values in losses
   - Verify improvement over baseline

## Error Recovery Protocol

When errors are detected:

1. **Stop Training**: Kill the process if it's stuck or erroring repeatedly
2. **Diagnose**: Read full error traceback, understand root cause
3. **Fix Code**: Modify the problematic module
4. **Test Fix**: Run minimal test to verify fix works
5. **Restart Training**: Only after confirming fix is working

## Debugging and Logging Best Practices

### Use logger.debug Instead of print

**IMPORTANT**: Always use `logger.debug()` instead of `print()` for debugging output in the HRR project. This ensures:

1. **Consistent Logging**: All debug messages go through the same logging system
2. **Log Level Control**: Debug messages can be enabled/disabled via configuration
3. **Structured Output**: Timestamps and module information are automatically included
4. **File Logging**: Debug messages are captured in log files for later analysis

#### Examples:

```python
# ❌ Don't use print
print(f"Loss components: {loss_dict}")

# ✅ Use logger.debug
logger.debug(f"Loss components: {loss_dict}")

# ✅ For important warnings
logger.warning(f"Zero loss detected at step {step}")

# ✅ For errors
logger.error(f"Failed to compute loss: {e}")
```

#### Logger Import:
```python
from hrr_diffusion.utils.logger import logger
```

## Project-Specific Considerations

### HRR Model Variants
- **SimpleHolographicViT**: Uses single Linear layers for heads
- **EnhancedHolographicViT**: Uses Sequential modules with normalization and dropout
- Visualization and analysis tools must handle both architectures

### Training Scripts
- `train_hrr_celeba.sh`: Standard model training
- `train_hrr_celeba_enhanced.sh`: Enhanced model with skip connections
- Both may have different hyperparameters and requirements

### Key Dependencies
- WandB for experiment tracking
- SDXL VAE for latent encoding
- CelebA dataset for training
- NMF for weight analysis

## Reporting Template

When reporting training results:

```
✅ Training Status:
- Model: [Standard/Enhanced]
- Epoch: X/Y
- Best Val Accuracy: XX.XX%
- Loss: X.XXXX
- WandB Errors: [None/List errors]
- Checkpoints Saved: [Yes/No]
- Next Steps: [...]
```

Always include error status explicitly in reports.