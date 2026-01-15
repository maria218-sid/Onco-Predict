import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load the cleaned data we saved on Friday
df = pd.read_csv("cleaned_cancer_data.csv")

# 2. AI GENERATED VIOLIN PLOT
# This code creates the basic structure of the plot
plt.figure(figsize=(10, 6))

# Logic: We compare 'diagnosis' on the X-axis and 'radius_mean' on the Y-axis
sns.violinplot(x='diagnosis', y='radius_mean', data=df, palette=['skyblue', 'salmon'])

# 3. YOUR TASK: MANUALLY TWEAK LABELS & TITLES
# Instruction: Update these strings to be more professional or descriptive
plt.title("Tumor Radius Comparison: Benign vs. Malignant")
plt.xlabel("Diagnosis (0 = Benign, 1 = Malignant)")
plt.ylabel("Scaled Radius Mean")

# Display the plot
plt.show()