from rnn.add_gate_layer import AddGateLayer
from rnn.multiple_gate_layer import MultipleGateLayer
from rnn.tanh_layer import TanhLayer


class RecurrentLayer(object):
    def __init__(self):
        self.mul_x_u = None
        self.mul_w_sp = None
        self.add_x_u_w_sp = None
        self.state = None
        self.mul_state_v = None

    def forward(self, input_data, state_previous, U, W, V):
        self.mul_x_u = MultipleGateLayer.forward(U, input_data)
        self.mul_w_sp = MultipleGateLayer.forward(W, state_previous)
        self.add_x_u_w_sp = AddGateLayer.forward(self.mul_x_u, self.mul_w_sp)
        self.state = TanhLayer.forward(self.add_x_u_w_sp)
        self.mul_state_v = MultipleGateLayer.forward(V, self.state)

    def backward(self, input_data, state_previous, U, W, V, loss_s, delta_mul_state_v):
        self.forward(input_data, state_previous, U, W, V)
        delta_v, delta_s_v = MultipleGateLayer.backward(V, self.state, delta_mul_state_v)
        delta_s = delta_s_v + loss_s
        delta_add_x_u_w_sp = TanhLayer.backward(self.add_x_u_w_sp, delta_s)
        delta_mul_w_sp, delta_mul_x_u = AddGateLayer.backward(self.mul_w_sp, self.mul_x_u, delta_add_x_u_w_sp)
        delta_w, delta_state_previous = MultipleGateLayer.backward(W, state_previous, delta_mul_w_sp)
        delta_u, delta_data = MultipleGateLayer.backward(U, input_data, delta_mul_x_u)
        return delta_state_previous, delta_u, delta_w, delta_v
