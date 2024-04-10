from .base import BaseNeck
from ..builder import NECKS


@NECKS.register()
class ModalSelection(BaseNeck):
    def __init__(self, modal: str = 'mix') -> None:
        super().__init__()

        self.modal = modal

    def forward(self, video=None, audio=None):
        if self.modal == 'video':
            return video
        elif self.modal == 'audio':
            return audio
        elif self.modal == 'mix':
            return {'video': video, 'audio': audio}
        else:
            raise NotImplementedError()
