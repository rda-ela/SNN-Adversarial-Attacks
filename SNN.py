import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision

import torchvision.transforms as transforms

from typing import NamedTuple

from norse.torch.functional.lif import LIFParameters
from norse.torch.module.leaky_integrator import LICell
from norse.torch.module.lif import LIFFeedForwardCell

import norse.torch.functional.encode as encode

import foolbox as fb
from foolbox import PyTorchModel, accuracy, samples
import foolbox.attacks as fa


batchsize = 64

transform = transforms.Compose(
    [transforms.Resize(32),
     transforms.ToTensor(),
     torchvision.transforms.Normalize((0.1307,), (0.3081,))])

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batchsize,
                                          shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batchsize,
                                         shuffle=False)

class IFConstantCurrentEncoder(torch.nn.Module):
    def __init__(
        self,
        seq_length,
        tau_mem_inv=1.0 / 1e-2,
        v_th=1.0,
        v_reset=0.0,
        dt: float = 0.001,
    ):
        super(IFConstantCurrentEncoder, self).__init__()
        self.seq_length = seq_length
        self.tau_mem_inv = tau_mem_inv
        self.v_th = v_th
        self.v_reset = v_reset
        self.dt = dt

    def forward(self, x):
        lif_parameters = LIFParameters(tau_mem_inv=self.tau_mem_inv, v_th=self.v_th, v_reset=self.v_reset)
        return encode.constant_current_lif_encode(x, self.seq_length, p=lif_parameters, dt=self.dt)

class ConvvNet4(torch.nn.Module):
    def __init__(
        self, device, num_channels=1, feature_size=32, method="super", dtype=torch.float
    ):
        super(ConvvNet4, self).__init__()
        self.features = int(((feature_size - 4) / 2 - 4) / 2)

        self.conv1 = torch.nn.Conv2d(1, 6, kernel_size=5, stride=1)
        self.conv2 = torch.nn.Conv2d(6, 16, kernel_size=5,stride=1)
        self.conv3 = torch.nn.Conv2d(16, 120, kernel_size=5, stride=1)
        self.fc1 = torch.nn.Linear(120, 84)

        self.lif0 = LIFFeedForwardCell(p=LIFParameters(method=method, alpha=100.0))
        self.lif1 = LIFFeedForwardCell(p=LIFParameters(method=method, alpha=100.0))
        self.lif2 = LIFFeedForwardCell(p=LIFParameters(method=method, alpha=100.0))
        self.lif3 = LIFFeedForwardCell(p=LIFParameters(method=method, alpha=100.0))
        self.out = LICell(84, 10)

        self.device = device
        self.dtype = dtype

    def forward(self, x):
        seq_length = x.shape[0]
        batch_size = x.shape[1]

        s0 = None
        s1 = None
        s2 = None
        s3 = None
        so = None

        voltages = torch.zeros(
            seq_length, batch_size, 10, device=self.device, dtype=self.dtype
        )

        for ts in range(seq_length):
            z = self.conv1(x[ts, :])
            z, s0 = self.lif0(z, s0)
            z = torch.nn.functional.max_pool2d(torch.nn.functional.relu(z), 2, 2)
            z = 10 * self.conv2(z)
            z, s1 = self.lif1(z, s1)
            z = torch.nn.functional.max_pool2d(torch.nn.functional.relu(z), 2, 2)
            z = 10 * self.conv3(z)
            z, s2 = self.lif2(z, s2)
            z = torch.nn.functional.relu(z)
            z = torch.flatten(z, 1)
            z = self.fc1(z)
            z, s3 = self.lif3(z, s3)
            v, so = self.out(torch.nn.functional.relu(z), so)
            voltages[ts, :, :] = v
        return voltages

        def train(model, device, train_loader, optimizer, epoch, writer=None):
            model.train()
            losses = []

            batch_len = len(train_loader)
            step = batch_len * epoch

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = torch.nn.functional.nll_loss(output, target)
                loss.backward()

                optimizer.step()
                step += 1

                if batch_idx % 100 == 0:
                    print(
                        "Train Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                            epoch,
                            epochs,
                            batch_idx * len(data),
                            len(train_loader.dataset),
                            100.0 * batch_idx / len(train_loader),
                            loss.item(),
                        )
                    )

                losses.append(loss.item())

            mean_loss = np.mean(losses)
            return losses, mean_loss


        def test(model, device, test_loader, epoch, writer=None):
            model.eval()
            test_loss = 0
            correct = 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    test_loss += torch.nn.functional.nll_loss(
                        output, target, reduction="sum"
                    ).item()
                    pred = output.argmax(
                        dim=1, keepdim=True
                    )
                    correct += pred.eq(target.view_as(pred)).sum().item()

            test_loss /= len(test_loader.dataset)

            accuracy = 100.0 * correct / len(test_loader.dataset)
            print(
                f"\nTest set {mmodel}: Average loss: {test_loss:.4f}, \
                    Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)\n"
            )

            return test_loss, accuracy

v_thh=[0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.25]

TT = [32,40,48,56,64,72,80]

for i in range(len(TT)):
    for j in range(len(v_thh)):

        print(TT[i],v_thh[j])

        class LIFConvNet(torch.nn.Module):
            def __init__(
                self,
                input_features,
                seq_length,
                model="super",
                only_first_spike=False,
            ):
                super(LIFConvNet, self).__init__()
                self.constant_current_encoder = IFConstantCurrentEncoder(seq_length=seq_length,v_th=v_thh[j])
                self.only_first_spike = only_first_spike
                self.input_features = input_features
                self.rsnn = ConvvNet4(method=model,device=device)
                self.seq_length = seq_length

            def forward(self, x):
                batch_size = x.shape[0]
                x = self.constant_current_encoder(
                    x.view(-1, self.input_features) * 1
                )
                if self.only_first_spike:
                    zeros = torch.zeros_like(x.cpu()).detach().numpy()
                    idxs = x.cpu().nonzero().detach().numpy()
                    spike_counter = np.zeros((batchsize, 32 * 32))
                    for t, batch, nrn in idxs:
                        if spike_counter[batch, nrn] == 0:
                            zeros[t, batch, nrn] = 1
                            spike_counter[batch, nrn] += 1
                    x = torch.from_numpy(zeros).to(x.device)

                x = x.reshape(self.seq_length, batch_size, 1, 32, 32)
                voltages = self.rsnn(x)
                m, _ = torch.max(voltages, 0)
                log_p_y = torch.nn.functional.log_softmax(m, dim=1)
                return log_p_y

        device = torch.device("cuda")
        mmodel = "super"
        lr = 0.001
        T = TT[i]
        input_features = 32 * 32

        model = LIFConvNet(
            input_features=input_features,
            seq_length=T,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        epochs = 1

        training_losses = []
        mean_losses = []
        test_losses = []
        accuracies = []

        for epoch in range(epochs):
                training_loss, mean_loss = train(model, device, train_loader, optimizer, epoch)
                test_loss, accuracy = test(model, device, test_loader, epoch)

                training_losses += training_loss
                mean_losses.append(mean_loss)
                test_losses.append(test_loss)
                accuracies.append(accuracy)

                max_accuracy = np.max(np.array(accuracies))

        adv_loader = torch.utils.data.DataLoader(testset, batch_size=batchsize,
                                                 shuffle=False)

        examples = enumerate(adv_loader)
        batch_idx, (images, labels) = next(examples)

        images = images.to(device)
        labels = labels.to(device)

        fmodel = fb.PyTorchModel(model, bounds=(0, 1))

        print("accuracy")
        print(accuracy(fmodel, images, labels))
        print("")

        attacks = [fb.attacks.PGD()]

        epsilons = [
            0.0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0,
            1.1,
            1.2,
            1.3,
            1.4,
            1.5,
            1.6,
            1.7,
            1.8,
            1.9,
            2.0,
        ]

        attack_success = np.zeros((len(attacks), len(epsilons), len(images)), dtype=np.bool)
        for i, attack in enumerate(attacks):
            _, _, success = attack(fmodel, images, labels, epsilons=epsilons)
            assert success.shape == (len(epsilons), len(images))
            success_ = success.cpu().numpy()
            assert success_.dtype == np.bool
            attack_success[i] = success_
            print(attack)
            print("  ", 1.0 - success_.mean(axis=-1).round(2))

        robust_accuracy = 1.0 - attack_success.max(axis=0).mean(axis=-1)
        print("")
        print("-" * 79)
        print("")
        print("worst case (best attack per-sample)")
        print("  ", robust_accuracy.round(2))
