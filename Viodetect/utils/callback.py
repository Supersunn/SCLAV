class DummyTrainer:
    def log(self, *args, **kwargs):
        pass

    @property
    def current_epoch(self):
        return 0


trainer = DummyTrainer()


def register_trainer(t):
    global trainer
    trainer = t


def get_trainer():
    return trainer
