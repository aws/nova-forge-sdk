# Nova Forge SDK — API Specification

## Contents

- [Service Classes](service-classes.md) — ForgeTrainer, ForgeEvaluator, ForgeDeployer, ForgeInference, ForgeConfig
- [NovaModelCustomizer](model-customizer.md) — Deprecated unified interface
- [Runtime Managers](runtime-managers.md) — SMTJ, SMHP, Glue runtime configuration
- [Dataset Loaders](dataset.md) — DatasetLoader, transforms, filters, operations
- [Job Results](job-results.md) — TrainingResult, EvalResult, DeployResult
- [Utilities](utilities.md) — Helper functions and monitoring
- [Enums and Configuration](enums.md) — Platform, Model, TrainingMethod, etc.
- [RFT Multiturn](rft-multiturn.md) — Multi-turn RFT infrastructure
- [Notifications](notifications.md) — Job notifications and error handling

---

## Best Practices
1. **Always validate your data** using `loader.show()` before training
2. **Use overrides sparingly** - start with defaults and tune as needed
3. **Monitor logs** during training using `get_logs()`
4. **Check job status** before calling `.get()` on results
5. **Clean up resources** when done to avoid unnecessary costs
6. **Use descriptive job names** to help track and organize your experiments
7. **Save results incrementally** during long-running jobs
8. **Test with small datasets** before scaling up to full training
---
## Additional Resources
- AWS Documentation: [Amazon Bedrock](https://docs.aws.amazon.com/bedrock/)
- AWS Documentation: [Amazon SageMaker](https://docs.aws.amazon.com/sagemaker/)
- SDK GitHub Repository: Check for updates and examples
- Support: Use AWS Support for technical assistance
---
