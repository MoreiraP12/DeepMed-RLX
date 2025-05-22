import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from src.tools.python_repl import python_repl_tool

def test_matplotlib_plot():
    # Test basic matplotlib plot
    code = """
import matplotlib.pyplot as plt
import numpy as np

# Create data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create plot
plt.figure(figsize=(8, 5))
plt.plot(x, y, 'b-', linewidth=2)
plt.title('Sine Wave')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.grid(True)

# Save the plot and print the IMAGE marker
plt.savefig('images/sine_wave.png')
print('IMAGE: images/sine_wave.png')
"""
    
    result = python_repl_tool(code)
    print(f"Result type: {type(result)}")
    print(f"Result: {result}")
    
    # Verify that the image file was created
    assert os.path.exists('images/sine_wave.png'), "Image file was not created"
    
    # Verify that the image path is in the result
    assert 'images/sine_wave.png' in result['images'], "Image path not found in result"
    
    print("Matplotlib test passed!")

def test_seaborn_plot():
    # Test seaborn plot
    code = """
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Create a dataset
data = np.random.normal(0, 1, 1000)
df = pd.DataFrame({'data': data})

# Create a plot
plt.figure(figsize=(8, 6))
sns.histplot(df['data'], kde=True)
plt.title('Histogram with KDE')
plt.xlabel('Value')
plt.ylabel('Frequency')

# Save the plot and print the IMAGE marker
plt.savefig('images/histogram.png')
print('IMAGE: images/histogram.png')
"""
    
    result = python_repl_tool(code)
    
    # Verify that the image file was created
    assert os.path.exists('images/histogram.png'), "Image file was not created"
    
    # Verify that the image path is in the result
    assert 'images/histogram.png' in result['images'], "Image path not found in result"
    
    print("Seaborn test passed!")

if __name__ == "__main__":
    # Create images directory if it doesn't exist
    os.makedirs('images', exist_ok=True)
    
    test_matplotlib_plot()
    test_seaborn_plot()
    
    print("All tests passed!") 