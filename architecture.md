# Model Architecture & Market Differentiation

## The Architecture
We are using an advanced deep learning technique called **Teacher-Student Knowledge Distillation**. 

*   **The Teacher (ResNet-50):** A massive, slow, highly accurate model (25+ million parameters). We train this first on our 5 combined global datasets to learn complex, deep hierarchical features of human lung anatomy, pneumonia, and global Tuberculosis variations.
*   **The Student (DenseNet-121):** A lightweight, highly efficient model (~8 million parameters). Instead of just learning from the raw images, the Student is trained to mimic the exact thought process (the "logits") of the Teacher. It learns how the Teacher makes decisions, retaining massive accuracy while remaining incredibly fast.
*   **The Phase C Domain Adaptation:** We use a unique final step where we freeze the model but unfreeze the `BatchNorm2d` layers, training it strictly on Indian (NIRT) data. This allows the model to shift its visual baseline to adapt to local Indian hardware.

## How it's different from the market
Most existing medical AI solutions (like those from Western companies) use single, massive models. They have two fatal flaws:
1. They require constant, high-speed internet to send images to a massive cloud GPU. 
2. They are trained on pristine Western datasets (like NIH). When introduced to an Indian hospital where X-ray machines might be older, have different contrasts, or where patients wear metal jewelry, those models fail.

**Our Difference:** Your Student model is so lightweight it can run entirely on a standard laptop without an internet connection. Furthermore, your Phase C training guarantees it actually understands and dynamically adjusts to the visual noise of Indian hospital scanners.
