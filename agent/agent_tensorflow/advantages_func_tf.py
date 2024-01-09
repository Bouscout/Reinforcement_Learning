# advantages function but for a tensorflow model
import tensorflow as tf
import keras.backend as K
import numpy as np

class GAE_estimations_tf:
    def __init__(self, baseline:object=None, gamma:float=0.99, _lambda_:float=0.95) -> None:
        # importing tensorflow only here

        import tensorflow as tf
        self.gamma = gamma
        self._lambda = _lambda_

        self.baseline = baseline

    def advantages(self, rewards, states, next_states ,dones, previous_values,loss_clip=0.2, batch_size=None, normalize=True):
        # in case we already pass in the values for value clipping
        mask = 1 - np.array(dones)
        mask = mask[:, None]

        if previous_values :
            values = previous_values
        else :
            values = self.baseline(states)

        next_values = self.baseline(next_states)

        td_error = rewards[:, None] + (self.gamma * next_values * mask) - values

        gaes = np.zeros_like(td_error)

        # calculating the advantages and the returns 
        for t in reversed(range(len(td_error) - 1)):
            gaes[t] = td_error[t] + (self.gamma * self._lambda * mask[t] * td_error[t + 1])

        target_values = gaes + values # for finding the gradient

        # checking if a batch size was given
        batch_size = batch_size if batch_size else len(td_error)

        index = 0
        for batch_index in range(batch_index, len(states), batch_size) :
            with tf.GradientTape() as Tape :
                loss_clip = loss_clip

                batch_states = states[index:batch_index]
                batch_target = target_values[index:batch_index]
                batch_previous_values = values[index:batch_index]

                estimations = self.baseline(batch_states)
                clipped_value = batch_previous_values + K.clip(estimations - batch_previous_values, -loss_clip, loss_clip)

                loss_1 = (batch_target - clipped_value)**2
                loss_2 = (batch_target - estimations)**2

                max_loss = K.maximum(loss_1, loss_2)

                loss = 0.5*K.mean(max_loss)

            gradient = Tape.gradient(loss, self.baseline.trainable_variables)

            self.baseline.optimizer.apply_gradients(zip(gradient, self.baseline.trainable_variables))

            index = batch_index

        # return the advantages
        if normalize :
            gaes -= gaes.mean()
            gaes /= gaes.std()

        return gaes
        
                
                


        
