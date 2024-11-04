import torch.nn as nn
import torchvision.models as models
from torch import Tensor
import yaml
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import datasets
from torchvision.transforms import v2
import torch
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import wandb
from tqdm import tqdm
from torch import GradScaler


class ResNet18CIFAR10(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18CIFAR10, self).__init__()

        self.model = models.resnet18(pretrained=False)
        print(self.model.fc.in_features)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


class PreActResNet18(nn.Module):
    def __init__(self, num_classes=100):
        super(PreActResNet18, self).__init__()

        self.layers = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Classifier
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: Tensor):
        return self.layers(x)


class Mlp(nn.Module):
    def __init__(self, input_size=28 * 28 * 3, hidden_size=256, num_classes=10):
        super(Mlp, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
            nn.LogSoftmax(dim=1)

        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)


class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.feature = nn.Sequential(
            # 1
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=2),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            # 2
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),

        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=16 * 5 * 5, out_features=120),
            nn.Tanh(),
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=10),
        )

    def forward(self, x):
        return self.classifier(self.feature(x))


class SimpleCachedDataset(Dataset):
    """Cache data for faster retrieval in training."""

    def __init__(self, dataset):
        # Pre-load all data into memory
        self.data = [item for item in dataset]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def load_model(config, sweep_config=None):
    if sweep_config:
        model_name = sweep_config.model
    else:
        model_name = config['model']['name']
    num_classes = config['model']['num_classes']

    if model_name == "resnet18_cifar10":
        return ResNet18CIFAR10(num_classes=num_classes)
    elif model_name == "preact_resnet18":
        return PreActResNet18(num_classes=num_classes)
    elif model_name == "mlp":
        return Mlp(num_classes=num_classes)
    elif model_name == "lenet":
        return LeNet(num_classes=num_classes)
    else:
        raise Exception(f"Unknown model")


class Trainer:
    def __init__(self, model, train_loader, val_loader, config, device, sweep_config=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.epochs = config['training']['epochs']
        self.patience = config['training']['early_stopping']
        self.criterion = nn.CrossEntropyLoss()
        if sweep_config:
            if sweep_config.optimizer == "adam":
                self.optimizer = torch.optim.Adam(model.parameters(), lr=sweep_config.learning_rate)
            elif sweep_config.optimizer == "sgd":
                self.optimizer = torch.optim.SGD(model.parameters(), lr=sweep_config.learning_rate, momentum=0.9,
                                                 nesterov=True)
            self.augmentation_type = sweep_config.augmentation_type
        else:
            self.augmentation_type = config['dataset']['augmentation']
            optim_name = config['training']['optimizer']
            lr = config['training']['learning_rate']
            self.optimizer = getattr(torch.optim, optim_name)(model.parameters(), lr=lr, momentum=0.9, nesterov=True)

        scheduler_name = config['training']['scheduler']
        if scheduler_name == "StepLR":
            self.scheduler = StepLR(self.optimizer, **config['training']['scheduler_params'])
        elif scheduler_name == "ReduceLROnPlateau":
            self.scheduler = ReduceLROnPlateau(self.optimizer)
        else:
            self.scheduler = None

        self.pin_memory = True
        self.enable_half = device == 'cuda'
        self.scaler = GradScaler(device, enabled=self.enable_half)

    def train(self):
        criterion = torch.nn.CrossEntropyLoss()
        self.model.train()
        correct = 0
        total = 0

        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
            with torch.autocast(self.device.type, enabled=self.enable_half):
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

            predicted = outputs.argmax(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        return 100.0 * correct / total

    @torch.inference_mode()
    def val(self):
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0
        for inputs, targets in self.val_loader:
            inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
            with torch.autocast(self.device.type, enabled=self.enable_half):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

            predicted = outputs.argmax(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            total_loss += loss

        return 100.0 * correct / total, total_loss

    @torch.inference_mode()
    def inference(self):
        self.model.eval()

        labels = []

        for inputs, _ in self.val_loader:
            inputs = inputs.to(self.device, non_blocking=True)
            with torch.autocast(self.device.type, enabled=self.enable_half):
                outputs = self.model(inputs)

            predicted = outputs.argmax(1).tolist()
            labels.extend(predicted)

        return labels

    def train_loop(self):
        best = 0.0
        epochs = list(range(self.epochs))
        best_val_loss = 1000
        counter = 0
        with tqdm(epochs) as tbar:
            for epoch in tbar:
                train_acc = self.train()
                val_acc, val_loss = self.val()
                if val_acc > best:
                    best = val_acc
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    counter = 0
                else:
                    counter += 1
                if counter == self.patience:
                    print("Stopping..")
                    break
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
                tbar.set_description(f"Train: {train_acc:.2f}, Val: {val_acc:.2f}, Best: {best:.2f}")
                wandb.log({"train_accuracy": train_acc, "epoch": epoch})
                wandb.log({"validation_accuracy": val_acc, "epoch": epoch})


def get_augmentation(augmentation_type):
    if augmentation_type == "flip":
        return v2.Compose([
            v2.RandomHorizontalFlip(),
            v2.ToDtype(torch.float32, scale=True)
        ])
    elif augmentation_type == "rotation":
        return v2.Compose([
            v2.RandomRotation((90, 270)),
            v2.ToDtype(torch.float32, scale=True)
        ])
    elif augmentation_type == "jitter":
        return v2.Compose([
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            v2.ToDtype(torch.float32, scale=True)
        ])
    else:
        return v2.ToDtype(torch.float32, scale=True)


def get_dataloader(config, sweep_config=None):
    if sweep_config:
        augmentation = get_augmentation(sweep_config.augmentation_type)
    else:
        augmentation = get_augmentation(config['dataset']['augmentation'])

    basic_transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25)),
    ])

    augmentation_transforms = v2.Compose([augmentation, basic_transforms])

    dataset_name = config['dataset']['name']
    dataset_path = config['dataset']['path']

    if dataset_name == "MNIST":
        train_dataset = datasets.MNIST(root=dataset_path, train=True, transform=basic_transforms, download=True)
        test_dataset = datasets.MNIST(root=dataset_path, train=False, transform=basic_transforms, download=True)
    elif dataset_name == "CIFAR10":
        train_dataset = datasets.CIFAR10(root=dataset_path, train=True, transform=basic_transforms, download=True)
        test_dataset = datasets.CIFAR10(root=dataset_path, train=False, transform=basic_transforms, download=True)
    elif dataset_name == "CIFAR100":
        train_dataset = datasets.CIFAR100(root=dataset_path, train=True, transform=basic_transforms, download=True)
        test_dataset = datasets.CIFAR100(root=dataset_path, train=False, transform=basic_transforms, download=True)
    else:
        raise ValueError(f"Unknown dataset")
    images = train_dataset.data
    labels = train_dataset.targets
    augmented_dataset = [(augmentation_transforms(img), label) for img, label in zip(images, labels)]

    train_dataset = SimpleCachedDataset(train_dataset)
    augmented_dataset = SimpleCachedDataset(augmented_dataset)
    test_dataset = SimpleCachedDataset(test_dataset)

    train_dataset = ConcatDataset([train_dataset, augmented_dataset])
    train_loader = DataLoader(train_dataset, batch_size=config['dataset']['batch_size'], shuffle=True,
                              pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config['dataset']['batch_size'], shuffle=False,
                             pin_memory=True)

    return train_loader, test_loader


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_sweep_config():
    return {
        'method': 'grid',
        'metric': {
            'name': 'validation_accuracy',
            'goal': 'maximize'
        },
        'parameters': {
            'augmentation_type': {
                'values': ['rotation', 'jitter']
            },
            'model': {
                'values': ['resnet18_cifar10', 'preact_resnet18']
            },
            'learning_rate': {
                'values': [0.0001, 0.01]
            },
            'optimizer': {
                'values': ['adam', 'sgd']
            }
        }
    }


def load_device(config):
    device = config['training']['device']
    if device == 'cuda' or device == 'gpu':
        if torch.cuda.is_available():
            return 'cuda'

    return 'cpu'


def main():
    run = wandb.init()
    model = load_model(config, run.config).to(device)
    train_loader, val_loader = get_dataloader(config, run.config)
    trainer = Trainer(model, train_loader, val_loader, config, device, run.config)
    trainer.train_loop()


wandb.login()
sweep_config = load_sweep_config()

config = load_config('/kaggle/input/dataset/config.yaml')
# config = load_config('config.yaml')
device = torch.device(load_device(config))

sweep_id = wandb.sweep(sweep=sweep_config, project="tema3")

wandb.agent(sweep_id, function=main, count=8)

wandb.finish()

# run without wandb
# model = load_model(config).to(device)
# train_loader, val_loader = get_dataloader(config)
# trainer = Trainer(model, train_loader, val_loader, config, device)
# trainer.train_loop()
