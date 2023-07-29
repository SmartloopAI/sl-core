import keras.callbacks


class TrainStatusReport(keras.callbacks.Callback):
    """prints '#' and reports training status
    Args:
        report_every: How many epochs between full reports
    """

    def __init__(self, report_every=10, current_epoch=0):
        super().__init__()
        self.report_every = report_every
        self.current_epoch = current_epoch

    def __report(self, epoch, logs=None):
        if self.current_epoch == 0:
            print(end='')
        else:
            print(end =' ')
        print('Epoch: {:d}, '.format(epoch), end='')
        for name, value in sorted(logs.items()):
            print('{}:{:0.4f}'.format(name, value), end=', ')
        print()

    def on_epoch_end(self, epoch, logs=None):
        self.current_epoch = epoch
        if epoch % self.report_every == 0:
            self.__report(self.current_epoch, logs)
        print('#', end='', flush=True)

    def on_train_end(self, logs=None):
        current = self.current_epoch + 1
        # print '#' to keep the formatting
        while current % self.report_every != 0:
            print('#', end='', flush=True)
            current += 1

        self.__report(self.current_epoch, logs)
        print()
