# creative-direction

This project uses mechanistic interpretability tools to explore how large language models internally represent the concept of “creativity” versus “formal” language. The goal was to investigate whether behaviors like creativity require higher-dimensional structure in a model’s activation space and whether linear methods could capture class differences.

### Approach

* Generated prompts instructing the model to use either creative or formal language.
* Collected residual stream activations from the Zephyr-7B-Beta model across multiple layers and token positions.
* Calculated the difference in mean activations between the two classes, then projected activations onto this direction to analyze linear separability.

### Key Insights

* Activations were nearly linearly separable between creative and formal instructions, suggesting that the model encodes different instruction types distinctly.
* Layer-wise analysis revealed that the model distinguishes instructions early in the processing pipeline.
* The original hypothesis about higher-dimensional structure was not supported, but the analysis did reveal some interesting insights into how a model processes instructions.
