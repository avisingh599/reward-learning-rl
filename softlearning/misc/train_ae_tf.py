from __future__ import absolute_import, division, print_function

# Import TensorFlow >= 1.9 and enable eager execution
import tensorflow as tf
tfe = tf.contrib.eager
tf.enable_eager_execution()

import os
import time
import numpy as np
import glob
import matplotlib.pyplot as plt
import PIL
import imageio
#from IPython import display
import os.path as osp

from softlearning.autoencoder.autoencoder_tf import VAE, SpatialAE


HDD = '/root/softlearning/data/'
ENV_NAME = 'sawyer_pusher_no_texture'
data_directory_experts = '/root/gym-larry/gym/envs/mujoco/assets/sawyer_pusher_data/expert_images_randomize_gripper_False_pos_noise_0.01_texture_False/task40/'


def compute_gradients(model, x):
    with tf.GradientTape() as tape:
        loss = model.compute_loss(x)
    return tape.gradient(loss, model.trainable_variables), loss

def apply_gradients(optimizer, gradients, variables, global_step=None):
    optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)

def generate_and_save_images(model, epoch, test_input, save_path):
  predictions = model.sample(test_input)
  fig = plt.figure(figsize=(4,4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, :])
      plt.axis('off')

  # tight_layout minimizes the overlap between 2 sub-plots
  plt.savefig(osp.join(save_path, 'image_at_epoch_{:04d}.png'.format(epoch)))
  #plt.show()

def main():

    data_directory = osp.join(HDD, 'random_trajectories', ENV_NAME)
    #data_path = osp.join(save_directory, 'combined_images.pkl')
    images = []
    file_list = sorted(os.listdir(data_directory_experts))
    for fname in file_list:
        if fname.endswith('.png'):
            image_path = os.path.join(data_directory_experts, fname)
            image = imageio.imread(image_path)
            image = image.astype(np.float32) / 255.
            images.append(image)
    file_list =  sorted(os.listdir(data_directory))
    for fname in file_list:
        if fname.endswith('.png'):
            image_path = os.path.join(data_directory, fname)
            image = imageio.imread(image_path)
            image = image.astype(np.float32) / 255.
            images.append(image)

    images = np.array(images)

    train_fraction = 0.9
    n_train = int(len(images) * train_fraction)
    indices = np.random.permutation(len(images))
    training_idx, test_idx = indices[:n_train], indices[n_train:]
    train_images, test_images = images[training_idx,:], images[test_idx,:]

    TRAIN_BUF = train_images.shape[0]
    TEST_BUF = test_images.shape[0]
    BATCH_SIZE = 128

    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(TRAIN_BUF).batch(BATCH_SIZE)
    test_dataset = tf.data.Dataset.from_tensor_slices(test_images).shuffle(TEST_BUF).batch(BATCH_SIZE)

    optimizer = tf.train.AdamOptimizer(1e-4)

    epochs = 100
    latent_dim = 32
    num_examples_to_generate = 16

    # keeping the random vector constant for generation (prediction) so
    # it will be easier to see the improvement.
    random_vector_for_generation = tf.random_normal(
        shape=[num_examples_to_generate, latent_dim])
    # model = VAE(latent_dim)
    model = SpatialAE(latent_dim, dropout=0.5)
    root = tf.train.Checkpoint(optimizer=optimizer,
                           model=model,
                           optimizer_step=tf.train.get_or_create_global_step())
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        for train_x in train_dataset:
            gradients, loss = compute_gradients(model, train_x)
            apply_gradients(optimizer, gradients, model.trainable_variables)
        end_time = time.time()

        if epoch % 1 == 0:
            loss = tfe.metrics.Mean()
            for test_x in test_dataset:
                loss(model.compute_loss(test_x))
            elbo = -loss.result()
            #display.clear_output(wait=False)
            print('Epoch: {}, Test set ELBO: {}, '
                    'time elapse for current epoch {}'.format(epoch,
                                                              elbo,
                                                              end_time - start_time))
            root.save('/root/ray_results/autoencoder_models_tf/spatialAE_dropout_{}'.format(epoch))

            # generate_and_save_images(
            #     model, epoch, random_vector_for_generation, '/root/ray_results/autoencoder_models_tf/')




if __name__ == "__main__":
    main()