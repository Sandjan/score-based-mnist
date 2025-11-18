import torch
import torch.nn.functional as F
from torch import nn


class ScoreNetwork0(torch.nn.Module):
    def __init__(self,num_classes=10,class_dim=3):
        super().__init__()
        chs = [32, 64, 128, 256, 256]
        act = torch.nn.LogSigmoid()
        self.class_encoder = torch.nn.Conv2d(num_classes, class_dim, kernel_size=1)
        self._convs = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Conv2d(2, chs[0], kernel_size=3, padding=1),  # (batch, ch, 28, 28)
                act,  # (batch, 8, 28, 28)
            ),
            torch.nn.Sequential(
                torch.nn.Conv2d(chs[0], chs[1], kernel_size=3, stride=2, padding=1),  # (batch, ch, 14, 14)
                act,  # (batch, 16, 14, 14)
            ),
            torch.nn.Sequential(
                torch.nn.Conv2d(chs[1], chs[2], kernel_size=3, stride=2, padding=1),  # (batch, ch, 7, 7)
                act,  # (batch, 32, 7, 7)
            ),
            torch.nn.Sequential(
                torch.nn.Conv2d(chs[2], chs[3], kernel_size=3, stride=2, padding=1),  # (batch, ch, 4, 4)
                act,  # (batch, 64, 4, 4)
            ),
            torch.nn.Sequential(
                torch.nn.Conv2d(chs[3], chs[4]-class_dim, kernel_size=3, stride=2, padding=1),  # (batch, ch, 2, 2)
                act,  # (batch, 64, 2, 2)
            ),
        ])
        self._tconvs = torch.nn.ModuleList([
                # input is the output of convs[4]
            torch.nn.Sequential(
                torch.nn.ConvTranspose2d(chs[4], chs[3], kernel_size=3, stride=2, padding=1, output_padding=1),  # (batch, 64, 4, 4)
                act,
            ),
            torch.nn.Sequential(
                # input is the output from the above sequential concated with the output from convs[3]
                torch.nn.ConvTranspose2d(chs[3]+64, chs[2], kernel_size=3, stride=2, padding=1, output_padding=0),  # (batch, 32, 7, 7)
                act,
            ),
            torch.nn.Sequential(
                # input is the output from the above sequential concated with the output from convs[2]
                torch.nn.ConvTranspose2d(chs[2]+64, chs[1], kernel_size=3, stride=2, padding=1, output_padding=1),  # (batch, chs[2], 14, 14)
                act,
            ),
            torch.nn.Sequential(
                # input is the output from the above sequential concated with the output from convs[1]
                torch.nn.ConvTranspose2d(chs[1]+64, chs[0], kernel_size=3, stride=2, padding=1, output_padding=1),  # (batch, chs[1], 28, 28)
                act,
            ),
            torch.nn.Sequential(
                # input is the output from the above sequential concated with the output from convs[0]
                torch.nn.Conv2d(chs[0]+32, chs[0], kernel_size=3, padding=1),  # (batch, chs[0], 28, 28)
                act,
                torch.nn.Conv2d(chs[0], 1, kernel_size=3, padding=1),  # (batch, 1, 28, 28)
            ),
        ])

    def forward(self, x,y, t) -> torch.Tensor:
        # takes an input image, class labels and time, returns the score function
        signal = torch.reshape(x, (*x.shape[:-1], 1, 28, 28))
        tt = t[..., None, None].expand(*t.shape[:-1], 1, signal.shape[-2], signal.shape[-1])
        signal = torch.cat((signal,tt), dim=-3)
        signals = []
        for i, conv in enumerate(self._convs):
            signal = conv(signal)
            if i < len(self._convs) - 1:
                signals.append(signal[:,:(32 if i==0 else 64),:,:])

        
        # encode and insert class conditioning on lowest level
        y_cond = y[:, :, None, None].float().expand(-1, -1, signal.shape[-2], signal.shape[-1])
        y_enc = torch.nn.functional.sigmoid(self.class_encoder(y_cond))
        signal = torch.cat((signal,y_enc), dim=-3)

        for i, tconv in enumerate(self._tconvs):
            if i == 0:
                signal = tconv(signal)
            else:
                signal = torch.cat((signal, signals[-i]), dim=-3)
                signal = tconv(signal)
        signal = torch.reshape(signal, (*signal.shape[:-3], -1))  # (..., 1 * 28 * 28)
        return signal