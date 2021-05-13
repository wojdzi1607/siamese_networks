from siamese_network import SiameseNetwork
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import tensorflow as tf

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# session = tf.Session(config=config)


def main():
    dataset_path = 'data/split_dataset_NOMASK/train'
    model_name = 'model_train_NOMASK'

    use_augmentation = True
    learning_rate = 1e-2
    batch_size = 16

    # Learning Rate multipliers for each layer
    learning_rate_multipliers = {'Conv1': 1, 'Conv2': 1, 'Conv3': 1, 'Conv4': 1, 'Dense1': 1}

    # l2-regularization penalization for each layer
    l2_penalization = {'Conv1': 1e-2, 'Conv2': 1e-2, 'Conv3': 1e-2, 'Conv4': 1e-2, 'Dense1': 1e-4}

    # Path where the logs will be saved
    tensorboard_log_path = f'./logs/{model_name}'
    siamese_network = SiameseNetwork(
        dataset_path=dataset_path,
        learning_rate=learning_rate,
        batch_size=batch_size, use_augmentation=use_augmentation,
        learning_rate_multipliers=learning_rate_multipliers,
        l2_regularization_penalization=l2_penalization,
        tensorboard_log_path=tensorboard_log_path
    )
    # Final layer-wise momentum (mu_j in the paper)
    momentum = 0.9
    # linear epoch slope evolution
    momentum_slope = 0.01
    support_set_size = 20
    evaluate_each = 500

    number_of_train_iterations = 30000

    validation_accuracy = siamese_network.train_siamese_network(number_of_iterations=number_of_train_iterations,
                                                                support_set_size=support_set_size,
                                                                final_momentum=momentum,
                                                                momentum_slope=momentum_slope,
                                                                evaluate_each=evaluate_each,
                                                                model_name=model_name)

    if validation_accuracy == 0:
        evaluation_accuracy = 0
    else:
        # Load the weights with best validation accuracy
        siamese_network.model.load_weights(f'./models/{model_name}.h5')
        evaluation_accuracy = siamese_network.omniglot_loader.one_shot_test(siamese_network.model, 20, 40, False)

    print('Final Evaluation Accuracy = ' + str(evaluation_accuracy))


if __name__ == "__main__":
    main()

