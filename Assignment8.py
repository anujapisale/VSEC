import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Load dataset
df = sns.load_dataset("penguins")
print(df.head(10))

# Remove rows containing missing values
df = df.dropna()

print(df.head(10))
print(df.info())

# Scatter plot
plt.figure(figsize=(7, 5))

sns.scatterplot(
    data=df,
    x="bill_length_mm",
    y="bill_depth_mm",
    hue="species",
    palette={
        "Adelie": "pink",
        "Chinstrap": "navy",
        "Gentoo": "red"
    }
)

plt.title("Scatter Plot: Bill Length vs. Bill Depth")
plt.xlabel("Bill Length (mm)")
plt.ylabel("Bill Depth (mm)")
plt.show()

# Correlation heatmap
plt.figure(figsize=(6, 5))

numeric_data = df.select_dtypes(include="number")
correlation_matrix = numeric_data.corr()

# Red = negative, White = zero, Green = positive
custom_cmap = LinearSegmentedColormap.from_list(
    "RedWhiteGreen",
    ["red", "white", "green"]
)

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap=custom_cmap,
    vmin=-1,
    vmax=1,
    fmt=".2f"
)

plt.title("Correlation Heatmap")
plt.show()
