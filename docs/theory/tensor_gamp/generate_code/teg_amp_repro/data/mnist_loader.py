import numpy as np
import os
import gzip
import urllib.request

def download_mnist(dest_dir='data/mnist'):
    """Downloads MNIST dataset if not already present."""
    base_url = 'http://yann.lecun.com/exdb/mnist/'
    files = [
        'train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte.gz',
        't10k-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz'
    ]
    
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        
    for file in files:
        file_path = os.path.join(dest_dir, file)
        if not os.path.exists(file_path):
            print(f"Downloading {file}...")
            try:
                urllib.request.urlretrieve(base_url + file, file_path)
            except Exception as e:
                print(f"Failed to download {file}: {e}")
                print("Using synthetic MNIST-like data as fallback.")
                return False
    return True

def load_mnist_images(file_path):
    """Loads MNIST images from a .gz file."""
    with gzip.open(file_path, 'rb') as f:
        # Read magic number and dimensions
        magic, num, rows, cols = np.frombuffer(f.read(16), dtype='>i4')
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
    return images

def get_mnist_tensor(num_digits=6, seed=None):
    """
    Creates a 28x28xN tensor by stacking MNIST digits.
    If MNIST cannot be downloaded, returns a synthetic tensor.
    """
    if seed is not None:
        np.random.seed(seed)
        
    mnist_dir = 'data/mnist'
    success = download_mnist(mnist_dir)
    
    if success:
        try:
            images = load_mnist_images(os.path.join(mnist_dir, 't10k-images-idx3-ubyte.gz'))
            # Normalize to [0, 1]
            images = images.astype(np.float64) / 255.0
            
            # Randomly select num_digits
            indices = np.random.choice(len(images), num_digits, replace=False)
            selected_images = images[indices]
            
            # Reshape to (28, 28, num_digits)
            tensor = np.transpose(selected_images, (1, 2, 0))
            return tensor
        except Exception as e:
            print(f"Error loading MNIST: {e}")
            
    # Fallback: Synthetic digit-like patterns
    print("Generating synthetic 28x28x6 tensor...")
    tensor = np.zeros((28, 28, num_digits))
    for i in range(num_digits):
        # Create some random rectangles/circles as "digits"
        x0, y0 = np.random.randint(5, 15, 2)
        w, h = np.random.randint(5, 10, 2)
        tensor[x0:x0+w, y0:y0+h, i] = 1.0
        # Add some noise
        tensor[:, :, i] += 0.1 * np.random.rand(28, 28)
        tensor[:, :, i] = np.clip(tensor[:, :, i], 0, 1)
        
    return tensor

def prepare_mnist_experiment(sampling_rate=0.4, seed=None):
    """
    Prepares data for the MNIST completion experiment.
    Returns:
        u_true: The original 28x28x6 tensor.
        y: The masked tensor.
        mask: The binary sampling mask.
    """
    u_true = get_mnist_tensor(num_digits=6, seed=seed)
    shape = u_true.shape
    
    if seed is not None:
        np.random.seed(seed)
        
    mask = (np.random.rand(*shape) < sampling_rate).astype(np.float64)
    y = u_true * mask
    
    return u_true, y, mask
