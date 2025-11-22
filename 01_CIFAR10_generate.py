from torchvision import datasets, transforms
import os
from PIL import Image
transform = transforms.ToTensor()

train_cifar10_dataset = datasets.CIFAR10(root='./dataset', train=True, download=True, transform=transform)
test_cifar10_dataset = datasets.CIFAR10(root='./dataset', train=False, download=True, transform=transform)

train_path = './dataset/CIFAR10/train'
test_path = './dataset/CIFAR10/test'

os.makedirs(train_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


print("Saving training images...")
for index, (image, label) in enumerate(train_cifar10_dataset):
    label_folder = os.path.join(train_path, class_names[label])
    os.makedirs(label_folder, exist_ok=True)
    image_path = os.path.join(label_folder, f'{index}.png')
    image = transforms.ToPILImage()(image)
    image.save(image_path)
    if index % 1000 == 0:
        print(f"{index} images saved...")


print("Saving test images...")
for index, (image, label) in enumerate(test_cifar10_dataset):
    label_folder = os.path.join(test_path, class_names[label])
    os.makedirs(label_folder, exist_ok=True)
    image_path = os.path.join(label_folder, f'{index}.png')
    image = transforms.ToPILImage()(image)
    image.save(image_path)
    if index % 1000 == 0:
        print(f"{index} images saved...")

print("All CIFAR-10 images have been saved successfully!")
