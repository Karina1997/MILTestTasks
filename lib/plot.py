import matplotlib.pyplot as plt
from IPython.display import clear_output

def plt_train(epoch, train_loss_epochs, eval_loss_epochs):
      clear_output(True)
      print('Epoch {0}... Train Loss: {1:.3f}'.format(
                        epoch, train_loss_epochs[-1]))
      print('Epoch {0}... Eval Loss: {1:.3f}'.format(
                        epoch, eval_loss_epochs[-1]))
      fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
      axes[1].plot(eval_loss_epochs)
      axes[0].plot(train_loss_epochs)
      fig.tight_layout()
      plt.setp(axes[0], xlabel='Epoch')
      plt.setp(axes[1], xlabel='Epoch')
      plt.setp(axes[0], ylabel='Loss')
                  

      axes[1].set_title('Dice loss validation')
      axes[0].set_title('Soft dice loss train')

      axes[0].grid()
      axes[1].grid()


      plt.show()