import torch
import config
import collections
import numpy as np
import time
import os

NetworkOutput = collections.namedtuple(
    'NetworkOutput',
    'value reward policy_logits hidden_state')


def dict_to_cpu(dictionary):
    cpu_dict = {}
    for key, value in dictionary.items():
        if isinstance(value, torch.Tensor):
            cpu_dict[key] = value.cpu()
        elif isinstance(value, dict):
            cpu_dict[key] = dict_to_cpu(value)
        else:
            cpu_dict[key] = value
    return cpu_dict


class AbstractNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def initial_inference(self, observation):
        pass

    def recurrent_inference(self, encoded_state, action):
        pass

    def get_weights(self):
        return dict_to_cpu(self.state_dict())

    def set_weights(self, weights):
        self.load_state_dict(weights)
        self.eval()

    # Renaming as to not override built-in functions
    def tft_save_model(self, episode):
        if not os.path.exists("./Checkpoints"):
            os.makedirs("./Checkpoints")

        path = f'./Checkpoints/checkpoint_{episode}'
        torch.save(self.state_dict(), path)

    # Renaming as to not override built-in functions
    def tft_load_model(self, episode):
        path = f'./Checkpoints/checkpoint_{episode}'
        if os.path.isfile(path):
            self.load_state_dict(torch.load(path))
            self.eval()
            print("Loading model episode {}".format(episode))
        else:
            print("Initializing model with new weights.")


class MuZeroNetwork(AbstractNetwork):
    def __init__(self):
        super().__init__()
        self.full_support_size = config.ENCODER_NUM_STEPS

        # self.representation_network = mlp(config.OBSERVATION_SIZE, [config.LAYER_HIDDEN_SIZE] *
        #                                   config.N_HEAD_HIDDEN_LAYERS, config.HIDDEN_STATE_SIZE)

        self.representation_network = RepresentationNetwork()

        # self.action_encodings = mlp(config.ACTION_CONCAT_SIZE, [config.LAYER_HIDDEN_SIZE] * 0,
        #                             config.HIDDEN_STATE_SIZE)
        
        self.dynamics_network = DynamicsNetwork()

        self.prediction_network = PredictionNetwork()

        self.value_encoder = ValueEncoder(*tuple(map(inverse_contractive_mapping, (-300., 300.))), 0)

        self.reward_encoder = ValueEncoder(*tuple(map(inverse_contractive_mapping, (-300., 300.))), 0)

    def prediction(self, encoded_state):
        policy, value = self.prediction_network(encoded_state)
        policy_logits = policy
        return policy_logits, value

    def representation(self, observation):
        observation = torch.from_numpy(observation).float().cuda()
        encoded_state = self.representation_network(observation)
        # Scale encoded state between [0, 1] (See appendix paper Training)
        min_encoded_state = encoded_state.min(1, keepdim=True)[0]
        max_encoded_state = encoded_state.max(1, keepdim=True)[0]
        scale_encoded_state = max_encoded_state - min_encoded_state
        scale_encoded_state[scale_encoded_state < 1e-5] += 1e-5
        encoded_state_normalized = (
                                           encoded_state - min_encoded_state
                                   ) / scale_encoded_state
        return encoded_state_normalized

    def dynamics(self, hidden_state, action):
        action = torch.from_numpy(action).to('cuda').to(torch.int64)
        one_hot_action = torch.nn.functional.one_hot(action[:, 0], config.ACTION_DIM[0])
        one_hot_target_a = torch.nn.functional.one_hot(action[:, 1], config.ACTION_DIM[1])
        one_hot_target_b = torch.nn.functional.one_hot(action[:, 2], config.ACTION_DIM[1])

        action_one_hot = torch.cat([one_hot_action, one_hot_target_a, one_hot_target_b], dim=-1).float()

        # action_encodings = self.action_encodings(action_one_hot)
        action_encodings = action_one_hot

        lstm_state = self.flat_to_lstm_input(hidden_state)

        inputs = action_encodings
        inputs = inputs[:, None, :]
        new_nested_states = []

        h0, c0 = list(zip(*lstm_state))

        state, reward_hidden, value_prefix = self.dynamics_network(inputs, (torch.stack(h0, dim=0), torch.stack(c0, dim=0)))

        # _, new_nested_states = self.dynamics_hidden_state_network(inputs, (torch.stack(h0, dim=0), torch.stack(c0, dim=0))) #guess state = new_states

        # print("SIZE", new_nested_states.size())
        next_hidden_state = self.rnn_to_flat(new_nested_states)  # (8, 1024) ##DOUBLE CHECK THIS

        # print("NEXT HIDDEN", next_hidden_state.size())
        reward = reward_hidden

        # Scale encoded state between [0, 1] (See paper appendix Training)
        min_next_hidden_state = next_hidden_state.min(1, keepdim=True)[0]
        max_next_hidden_state = next_hidden_state.max(1, keepdim=True)[0]
        scale_next_hidden_state = max_next_hidden_state - min_next_hidden_state
        scale_next_hidden_state[scale_next_hidden_state < 1e-5] += 1e-5
        next_hidden_state_normalized = (
                                                next_hidden_state - min_next_hidden_state
                                        ) / scale_next_hidden_state

        return next_hidden_state_normalized, reward

    def initial_inference(self, observation):
        hidden_state = self.representation(observation)
        policy_logits, value_logits = self.prediction(hidden_state)

        reward = np.zeros(observation.shape[0])

        value = self.value_encoder.decode(torch.softmax(value_logits, dim=-1).detach().cpu().numpy())
        reward_logits = self.reward_encoder.encode(reward)

        outputs = {
            "value": value,
            "value_logits": value_logits,
            "reward": reward,
            "reward_logits": reward_logits,
            "policy_logits": policy_logits,
            "hidden_state": hidden_state
        }
        return outputs

    @staticmethod
    def rnn_to_flat(state):
        """Maps LSTM state to flat vector."""
        states = []
        for cell_state in state:
            states.extend(cell_state)
        return torch.cat(states, dim=-1)

    @staticmethod
    def flat_to_lstm_input(state):
        """Maps flat vector to LSTM state."""
        tensors = []
        cur_idx = 0
        for size in config.RNN_SIZES:
            states = (state[Ellipsis, cur_idx:cur_idx + size],
                      state[Ellipsis, cur_idx + size:cur_idx + 2 * size])

            cur_idx += 2 * size
            tensors.append(states)
        # assert cur_idx == state.shape[-1]
        return tensors

    def recurrent_inference(self, hidden_state, action):
        next_hidden_state, reward_logits = self.dynamics(hidden_state, action)
        policy_logits, value_logits = self.prediction(next_hidden_state)

        value = self.value_encoder.decode(torch.softmax(value_logits, dim=-1).detach().cpu().numpy())
        reward = self.reward_encoder.decode(torch.softmax(reward_logits, dim=-1).detach().cpu().numpy())

        outputs = {
            "value": value,
            "value_logits": value_logits,
            "reward": reward,
            "reward_logits": reward_logits,
            "policy_logits": policy_logits,
            "hidden_state": next_hidden_state
        }
        return outputs

def mlp(
    input_size,
    layer_sizes,
    output_size,
    output_activation=torch.nn.Identity,
    activation=torch.nn.ReLU,
    momentum=0.1,
    init_zero=False,
):
    """MLP layers
    Parameters
    ----------
    input_size: int
        dim of inputs
    layer_sizes: list
        dim of hidden layers
    output_size: int
        dim of outputs
    init_zero: bool
        zero initialization for the last layer (including w and b).
        This can provide stable zero outputs in the beginning.
    """
    sizes = [input_size] + layer_sizes + [output_size]
    layers = []
    for i in range(len(sizes) - 1):
        if i < len(sizes) - 2:
            act = activation
            layers += [torch.nn.Linear(sizes[i], sizes[i + 1]),
                       torch.nn.BatchNorm1d(sizes[i + 1], momentum=momentum),
                       act()]
        else:
            act = output_activation
            layers += [torch.nn.Linear(sizes[i], sizes[i + 1]),
                       act()]

    if init_zero:
        layers[-2].weight.data.fill_(0)
        layers[-2].bias.data.fill_(0)

    return torch.nn.Sequential(*layers)

# Linear(input, layer_size) -> RELU
#      -> Linear -> Identity -> 0
#      -> Linear -> Identity -> 1
#      ... for each size in output_size
#  -> output -> [0, 1, ... n]
class MultiMlp(torch.nn.Module):
    def __init__(self,
                 input_size,
                 layer_size,
                 output_sizes,
                 output_activation=torch.nn.Identity,
                 activation=torch.nn.LeakyReLU):
        super().__init__()

        layers = []
        layers += [torch.nn.Linear(input_size, layer_size[0]), activation()]
        for i in range(len(layer_size) - 1):
            layers += [torch.nn.Linear(layer_size[i], layer_size[i + 1]), activation()]
        self.encoding_layer = torch.nn.Sequential(*layers).cuda()

        # self.output_heads = []

        self.head_0 = torch.nn.Sequential(
                torch.nn.Linear(layer_size[-1], output_sizes[0]),
                output_activation()
            ).cuda()
        
        self.head_1 = torch.nn.Sequential(
                torch.nn.Linear(layer_size[-1], output_sizes[1]),
                output_activation()
            ).cuda()
        
        self.head_2 = torch.nn.Sequential(
                torch.nn.Linear(layer_size[-1], output_sizes[2]),
                output_activation()
            ).cuda()
        
        self.head_3 = torch.nn.Sequential(
                torch.nn.Linear(layer_size[-1], output_sizes[3]),
                output_activation()
            ).cuda()
        
        self.head_4 = torch.nn.Sequential(
                torch.nn.Linear(layer_size[-1], output_sizes[4]),
                output_activation()
            ).cuda()

        # for size in output_sizes:
        #     output_layer = torch.nn.Sequential(
        #         torch.nn.Linear(layer_size[-1], size),
        #         output_activation()
        #     ).cuda()
        #     self.output_heads.append(output_layer)

    def forward(self, x):
        # Encode the hidden state
        x = self.encoding_layer(x)

        # Pass x into all output heads
        output = []

        output.append(self.head_0(x))
        output.append(self.head_1(x))
        output.append(self.head_2(x))
        output.append(self.head_3(x))
        output.append(self.head_4(x))

        # return torch.cat(output, dim=-1)
        return output

    def __call__(self, x):
        return self.forward(x)
    
def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> torch.nn.Conv2d:
    """1x1 convolution"""
    return torch.nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_channels, out_channels, stride=1):
    return torch.nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
    )

# Residual block
class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, downsample=None, stride=1, momentum=0.1):
        super().__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = torch.nn.BatchNorm2d(out_channels, momentum=momentum)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = torch.nn.BatchNorm2d(out_channels, momentum=momentum)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.nn.functional.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = torch.nn.functional.relu(out)
        return out

class DownSample(torch.nn.Module):
    def __init__(self, in_channels, out_channels, momentum=0.1):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels,
            out_channels // 2,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.bn1 = torch.nn.BatchNorm2d(out_channels // 2, momentum=momentum)
        self.resblocks1 = torch.nn.ModuleList(
            [ResidualBlock(out_channels // 2, out_channels // 2, momentum=momentum) for _ in range(1)]
        )
        self.conv2 = torch.nn.Conv2d(
            out_channels // 2,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.downsample_block = ResidualBlock(out_channels // 2, out_channels, momentum=momentum, stride=2, downsample=self.conv2)
        self.resblocks2 = torch.nn.ModuleList(
            [ResidualBlock(out_channels, out_channels, momentum=momentum) for _ in range(1)]
        )
        self.pooling1 = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.resblocks3 = torch.nn.ModuleList(
            [ResidualBlock(out_channels, out_channels, momentum=momentum) for _ in range(1)]
        )
        self.pooling2 = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.nn.functional.relu(x)
        for block in self.resblocks1:
            x = block(x)
        x = self.downsample_block(x)
        for block in self.resblocks2:
            x = block(x)
        x = self.pooling1(x)
        for block in self.resblocks3:
            x = block(x)
        x = self.pooling2(x)
        return x

# Encode the observations into hidden states
class RepresentationNetwork(torch.nn.Module):
    def __init__(
        self,
        observation_shape,
        num_blocks,
        num_channels,
        downsample,
        momentum=0.1,
    ):
        """Representation network
        Parameters
        ----------
        observation_shape: tuple or list
            shape of observations: [C, W, H]
        num_blocks: int
            number of res blocks
        num_channels: int
            channels of hidden states
        downsample: bool
            True -> do downsampling for observations. (For board games, do not need)
        """
        super().__init__()
        self.downsample = downsample
        if self.downsample:
            self.downsample_net = DownSample(
                observation_shape[0],
                num_channels,
            )
        self.conv = conv3x3(
            observation_shape[0],
            num_channels,
        )
        self.bn = torch.nn.BatchNorm2d(num_channels, momentum=momentum)
        self.resblocks = torch.nn.ModuleList(
            [ResidualBlock(num_channels, num_channels, momentum=momentum) for _ in range(num_blocks)]
        )

    def forward(self, x):
        if self.downsample:
            x = self.downsample_net(x)
        else:
            x = self.conv(x)
            x = self.bn(x)
            x = torch.nn.functional.relu(x)

        for block in self.resblocks:
            x = block(x)
        return x

    def get_param_mean(self):
        mean = []
        for name, param in self.named_parameters():
            mean += np.abs(param.detach().cpu().numpy().reshape(-1)).tolist()
        mean = sum(mean) / len(mean)
        return mean

# Predict next hidden states given current states and actions
class DynamicsNetwork(torch.nn.Module):
    def __init__(
        self,
        num_blocks,
        num_channels,
        reduced_channels_reward,
        fc_reward_layers,
        full_support_size,
        block_output_size_reward,
        lstm_hidden_size=64,
        momentum=0.1,
        init_zero=False,
    ):
        """Dynamics network
        Parameters
        ----------
        num_blocks: int
            number of res blocks
        num_channels: int
            channels of hidden states
        fc_reward_layers: list
            hidden layers of the reward prediction head (MLP head)
        full_support_size: int
            dim of reward output
        block_output_size_reward: int
            dim of flatten hidden states
        lstm_hidden_size: int
            dim of lstm hidden
        init_zero: bool
            True -> zero initialization for the last layer of reward mlp
        """
        super().__init__()
        self.num_channels = num_channels
        self.lstm_hidden_size = lstm_hidden_size

        self.conv = conv3x3(num_channels, num_channels - 1)
        self.bn = torch.nn.BatchNorm2d(num_channels - 1, momentum=momentum)
        self.resblocks = torch.nn.ModuleList(
            [ResidualBlock(num_channels - 1, num_channels - 1, momentum=momentum) for _ in range(num_blocks)]
        )

        self.reward_resblocks = torch.nn.ModuleList(
            [ResidualBlock(num_channels - 1, num_channels - 1, momentum=momentum) for _ in range(num_blocks)]
        )

        self.conv1x1_reward = torch.nn.Conv2d(num_channels - 1, reduced_channels_reward, 1)
        self.bn_reward = torch.nn.BatchNorm2d(reduced_channels_reward, momentum=momentum)
        self.block_output_size_reward = block_output_size_reward
        self.lstm = torch.nn.LSTM(input_size=self.block_output_size_reward, hidden_size=self.lstm_hidden_size)
        self.bn_value_prefix = torch.nn.BatchNorm1d(self.lstm_hidden_size, momentum=momentum)
        self.fc = mlp(self.lstm_hidden_size, fc_reward_layers, full_support_size, init_zero=init_zero, momentum=momentum)

    def forward(self, x, reward_hidden):
        state = x[:,:-1,:,:]
        x = self.conv(x)
        x = self.bn(x)

        x += state
        x = torch.nn.functional.relu(x)

        for block in self.resblocks:
            x = block(x)
        state = x

        x = self.conv1x1_reward(x)
        x = self.bn_reward(x)
        x = torch.nn.functional.relu(x)

        x = x.view(-1, self.block_output_size_reward).unsqueeze(0)
        value_prefix, reward_hidden = self.lstm(x, reward_hidden)
        value_prefix = value_prefix.squeeze(0)
        value_prefix = self.bn_value_prefix(value_prefix)
        value_prefix = torch.nn.functional.relu(value_prefix)
        value_prefix = self.fc(value_prefix)

        return state, reward_hidden, value_prefix

    def get_dynamic_mean(self):
        dynamic_mean = np.abs(self.conv.weight.detach().cpu().numpy().reshape(-1)).tolist()

        for block in self.resblocks:
            for name, param in block.named_parameters():
                dynamic_mean += np.abs(param.detach().cpu().numpy().reshape(-1)).tolist()
        dynamic_mean = sum(dynamic_mean) / len(dynamic_mean)
        return dynamic_mean

    def get_reward_mean(self):
        reward_w_dist = self.conv1x1_reward.weight.detach().cpu().numpy().reshape(-1)

        for name, param in self.fc.named_parameters():
            temp_weights = param.detach().cpu().numpy().reshape(-1)
            reward_w_dist = np.concatenate((reward_w_dist, temp_weights))
        reward_mean = np.abs(reward_w_dist).mean()
        return reward_w_dist, reward_mean

# predict the value and policy given hidden states
class PredictionNetwork(torch.nn.Module):
    def __init__(
        self,
        action_space_size,
        num_blocks,
        num_channels,
        reduced_channels_value,
        reduced_channels_policy,
        fc_value_layers,
        fc_policy_layers,
        full_support_size,
        block_output_size_value,
        block_output_size_policy,
        momentum=0.1,
        init_zero=False,
    ):
        """Prediction network
        Parameters
        ----------
        action_space_size: int
            action space
        num_blocks: int
            number of res blocks
        num_channels: int
            channels of hidden states
        reduced_channels_value: int
            channels of value head
        reduced_channels_policy: int
            channels of policy head
        fc_value_layers: list
            hidden layers of the value prediction head (MLP head)
        fc_policy_layers: list
            hidden layers of the policy prediction head (MLP head)
        full_support_size: int
            dim of value output
        block_output_size_value: int
            dim of flatten hidden states
        block_output_size_policy: int
            dim of flatten hidden states
        init_zero: bool
            True -> zero initialization for the last layer of value/policy mlp
        """
        super().__init__()
        self.resblocks = torch.nn.ModuleList(
            [ResidualBlock(num_channels, num_channels, momentum=momentum) for _ in range(num_blocks)]
        )

        self.conv1x1_value = torch.nn.Conv2d(num_channels, reduced_channels_value, 1)
        self.conv1x1_policy = torch.nn.Conv2d(num_channels, reduced_channels_policy, 1)
        self.bn_value = torch.nn.BatchNorm2d(reduced_channels_value, momentum=momentum)
        self.bn_policy = torch.nn.BatchNorm2d(reduced_channels_policy, momentum=momentum)
        self.block_output_size_value = block_output_size_value
        self.block_output_size_policy = block_output_size_policy
        self.fc_value = mlp(self.block_output_size_value, fc_value_layers, full_support_size, init_zero=init_zero, momentum=momentum)
        self.fc_policy = mlp(self.block_output_size_policy, fc_policy_layers, action_space_size, init_zero=init_zero, momentum=momentum)

    def forward(self, x):
        for block in self.resblocks:
            x = block(x)
        value = self.conv1x1_value(x)
        value = self.bn_value(value)
        value = torch.nn.functional.relu(value)

        policy = self.conv1x1_policy(x)
        policy = self.bn_policy(policy)
        policy = torch.nn.functional.relu(policy)

        value = value.view(-1, self.block_output_size_value)
        policy = policy.view(-1, self.block_output_size_policy)
        value = self.fc_value(value)
        policy = self.fc_policy(policy)
        return policy, value

class ValueEncoder:
    """Encoder for reward and value targets from Appendix of MuZero Paper."""

    def __init__(self,
                 min_value,
                 max_value,
                 num_steps,
                 use_contractive_mapping=True):
        if not max_value > min_value:
            raise ValueError('max_value must be > min_value')
        min_value = float(min_value)
        max_value = float(max_value)
        if use_contractive_mapping:
            max_value = contractive_mapping(max_value)
            min_value = contractive_mapping(min_value)
        if num_steps <= 0:
            num_steps = np.ceil(max_value) + 1 - np.floor(min_value)
        self.min_value = min_value
        self.max_value = max_value
        self.value_range = max_value - min_value
        self.num_steps = num_steps
        self.step_size = self.value_range / (num_steps - 1)
        self.step_range_int = np.arange(0, self.num_steps, dtype=int)
        self.step_range_float = self.step_range_int.astype(float)
        self.use_contractive_mapping = use_contractive_mapping

    def encode(self, value):  # not worth optimizing
        if len(value.shape) != 1:
            raise ValueError(
                'Expected value to be 1D Tensor [batch_size], but got {}.'.format(
                    value.shape))
        if self.use_contractive_mapping:
            value = contractive_mapping(value)
        value = np.expand_dims(value, -1)
        clipped_value = np.clip(value, self.min_value, self.max_value)
        above_min = clipped_value - self.min_value
        num_steps = above_min / self.step_size
        lower_step = np.floor(num_steps)
        upper_mod = num_steps - lower_step
        lower_step = lower_step.astype(int)
        upper_step = lower_step + 1
        lower_mod = 1.0 - upper_mod
        lower_encoding, upper_encoding = (
            np.equal(step, self.step_range_int).astype(float) * mod
            for step, mod in (
                (lower_step, lower_mod),
                (upper_step, upper_mod),)
        )
        return lower_encoding + upper_encoding

    def decode(self, logits):  # not worth optimizing
        if len(logits.shape) != 2:
            raise ValueError(
                'Expected logits to be 2D Tensor [batch_size, steps], but got {}.'
                .format(logits.shape))
        num_steps = np.sum(logits * self.step_range_float, -1)
        above_min = num_steps * self.step_size
        value = above_min + self.min_value
        if self.use_contractive_mapping:
            value = inverse_contractive_mapping(value)
        return value


# From the MuZero paper.
def contractive_mapping(x, eps=0.001):
    return np.sign(x) * (np.sqrt(np.abs(x) + 1.) - 1.) + eps * x


# From the MuZero paper.
def inverse_contractive_mapping(x, eps=0.001):
    return np.sign(x) * \
           (np.square((np.sqrt(4 * eps * (np.abs(x) + 1. + eps) + 1.) - 1.) / (2. * eps)) - 1.)

# Softmax function in np because we're converting it anyway
def softmax_stable(x):
    return np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum()

