"�I
DDeviceIDLE"IDLE1�����ƠBA�����ƠBQ      �?Y      �?�Unknown
BHostIDLE"IDLE1fffff�@Afffff�@a��QP�,�?i��QP�,�?�Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1������@9������@A������@I������@aa�h#R��?i�b	��?�Unknown�
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1�����c@9�����c@A�����c@I�����c@a�Ŧd��?iP0<����?�Unknown
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1�����\b@9�����\b@A�����\b@I�����\b@a������?i��4��?�Unknown
qHost_FusedMatMul"sequential/dense_1/Relu(1fffffFW@9fffffFW@AfffffFW@IfffffFW@a�he��]�?ip��/c �?�Unknown
dHostDataset"Iterator::Model(1�����S@9�����S@A�����,Q@I�����,Q@a!dhG%�?i�-8M�\�?�Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(1�����Q@9�����Q@A�����Q@I�����Q@ag�5�'��?i�����?�Unknown
o	Host_FusedMatMul"sequential/dense/Relu(133333�L@933333�L@A33333�L@I33333�L@a�Fˁ�?i�e���?�Unknown
o
HostSoftmax"sequential/dense_2/Softmax(1      C@9      C@A      C@I      C@a��l��y?i�?@�:�?�Unknown
�HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1������7@9������7@A������7@I������7@ah�C5��o?i��u��Y�?�Unknown
tHost_FusedMatMul"sequential/dense_2/BiasAdd(1������6@9������6@A������6@I������6@a�aң�tn?i�U�`x�?�Unknown
^HostGatherV2"GatherV2(1�����L6@9�����L6@A�����L6@I�����L6@a��`n?i�+ n��?�Unknown
{HostMatMul"'gradient_tape/sequential/dense_2/MatMul(1������5@9������5@A������5@I������5@a^*��`m?i�0��γ�?�Unknown
}HostMatMul")gradient_tape/sequential/dense_2/MatMul_1(1ffffff1@9ffffff1@Affffff1@Iffffff1@a�x��rg?iu
�A��?�Unknown
�Host#SparseSoftmaxCrossEntropyWithLogits"gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits(1������.@9������.@A������.@I������.@a�m���d?i�"����?�Unknown
}HostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1ffffff-@9ffffff-@Affffff-@Iffffff-@a']��j�c?i@�����?�Unknown
�HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1ffffff.@9ffffff.@Affffff(@Iffffff(@a}���p`?i��xC�?�Unknown
�HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1      /@9      /@A333333'@I333333'@a�K���C_?iu�����?�Unknown
iHostWriteSummary"WriteSummary(1ffffff%@9ffffff%@Affffff%@Iffffff%@a���[��\?i���kP"�?�Unknown�
�HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1������@9������@A������@I������@a�	h`�JU?i��:��,�?�Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1������@9������@A������@I������@a�m���T?ixE4UV7�?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1333333@9333333@A333333@I333333@a��Ԇ�SR?iЯwK�@�?�Unknown
HostReluGrad")gradient_tape/sequential/dense_1/ReluGrad(1      @9      @A      @I      @a��b�Q?i�)�BI�?�Unknown
�HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1������@9������@A������@I������@a+y��?Q?iq���Q�?�Unknown
�HostMul"Ugradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul(1333333@9333333@A333333@I333333@a�K���CO?iD���Y�?�Unknown
�HostSum"1sparse_categorical_crossentropy/weighted_loss/Sum(1333333@9333333@A333333@I333333@a�K���CO?ic��a�?�Unknown
`HostGatherV2"
GatherV2_1(1������@9������@A������@I������@ay�`�M?iN����h�?�Unknown
ZHostArgMax"ArgMax(1333333@9333333@A333333@I333333@a�@ ��L?i^��o�?�Unknown
gHostStridedSlice"strided_slice(1������@9������@A������@I������@a椟7�L?iG͞�v�?�Unknown
�HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1333333@9333333@A333333@I333333@a�5���I?i�T�
j}�?�Unknown
l HostIteratorGetNext"IteratorGetNext(1������@9������@A������@I������@a����UI?i������?�Unknown
�!HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1333333@9333333@A333333@I333333@a�*:_�-G?iER�����?�Unknown
�"HostPack"/sparse_categorical_crossentropy/Reshape_1/shape(1333333@9333333@A333333@I333333@a�*:_�-G?i� �wV��?�Unknown
�#HostBiasAddGrad"4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGrad(1ffffff@9ffffff@Affffff@Iffffff@a?�x��F?i?��ܔ�?�Unknown
�$HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1333333@9333333@A333333@I333333@a�����E?i��?l��?�Unknown
�%HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1333333@9333333@A333333@I333333@a�����E?i���_��?�Unknown
x&HostDataset"#Iterator::Model::ParallelMapV2::Zip(1������A@9������A@A������@I������@a�����C?i��]a\��?�Unknown
|'HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1333333@9333333@A333333@I333333@a��Ԇ�SB?i��\��?�Unknown
�(HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9      �?A      @I      �?a�AR��+@?iH�3X���?�Unknown
X)HostCast"Cast_1(1333333@9333333@A333333@I333333@a�K���C??i�����?�Unknown
�*HostTile"Bgradient_tape/sparse_categorical_crossentropy/weighted_loss/Tile_1(1333333@9333333@A333333@I333333@a�K���C??i��Oʹ�?�Unknown
e+Host
LogicalAnd"
LogicalAnd(1ffffff@9ffffff@Affffff@Iffffff@a"��/>?i]�K���?�Unknown�
�,HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1ffffff@9ffffff@Affffff@Iffffff@a"��/>?i�#�GY��?�Unknown
V-HostCast"Cast(1������@9������@A������@I������@ay�`�=?i�o����?�Unknown
b.HostDivNoNan"div_no_nan_1(1������@9������@A������@I������@ay�`�=?iػ%@���?�Unknown
�/HostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1������@9������@A������@I������@a椟7�<?iͯl<!��?�Unknown
�0HostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1������@9������@A������@I������@a椟7�<?i£�8���?�Unknown
�1HostCast"bsparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast_1(1������@9������@A������@I������@a椟7�<?i���4#��?�Unknown
X2HostEqual"Equal(1      @9      @A      @I      @aSm�\��:?i�3f����?�Unknown
t3HostAssignAddVariableOp"AssignAddVariableOp(1ffffff@9ffffff@Affffff@Iffffff@a.�[���8?i.���?�Unknown
�4HostReadVariableOp"(sequential/dense_2/MatMul/ReadVariableOp(1ffffff@9ffffff@Affffff@Iffffff@a.�[���8?i�
Ъ���?�Unknown
v5HostAssignAddVariableOp"AssignAddVariableOp_4(1������@9������@A������@I������@a�ƚ��7?iޝ�����?�Unknown
V6HostSum"Sum_2(1������@9������@A������@I������@a�ƚ��7?i71�����?�Unknown
X7HostCast"Cast_3(1������ @9������ @A������ @I������ @a	����6?iil�!w��?�Unknown
�8HostStridedSlice"-sparse_categorical_crossentropy/strided_slice(1������ @9������ @A������ @I������ @a	����6?i���K��?�Unknown
�9HostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1�������?9�������?A�������?I�������?aQ�a�g3?iX�뛸��?�Unknown
w:HostReadVariableOp"div_no_nan_1/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?a��Ԇ�S2?i�|��?�Unknown
X;HostCast"Cast_2(1      �?9      �?A      �?I      �?a�AR��+0?i6�V���?�Unknown
�<HostDivNoNan"Lgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/div_no_nan(1      �?9      �?A      �?I      �?a�AR��+0?i~	1��?�Unknown
v=HostAssignAddVariableOp"AssignAddVariableOp_2(1�������?9�������?A�������?I�������?a椟7�,?ix�T����?�Unknown
�>HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1������0@9������0@A�������?I�������?a椟7�,?ir�w���?�Unknown
T?HostMul"Mul(1333333�?9333333�?A333333�?I333333�?a�5���)?iE�-��?�Unknown
s@HostReadVariableOp"SGD/Cast/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?a�5���)?iA���?�Unknown
`AHostDivNoNan"
div_no_nan(1333333�?9333333�?A333333�?I333333�?a�5���)?i�bPi��?�Unknown
XBHostCast"Cast_4(1�������?9�������?A�������?I�������?a�ƚ��'?i�,�����?�Unknown
�CHostDivNoNan"3sparse_categorical_crossentropy/weighted_loss/value(1�������?9�������?A�������?I�������?a�ƚ��'?iC�)	`��?�Unknown
vDHostAssignAddVariableOp"AssignAddVariableOp_3(1�������?9�������?A�������?I�������?aQ�a�g#?i������?�Unknown
�EHostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1�������?9�������?A�������?I�������?aQ�a�g#?i)����?�Unknown
wFHostReadVariableOp"div_no_nan/ReadVariableOp_1(1�������?9�������?A�������?I�������?a+y��?!?i9�p���?�Unknown
�GHostReadVariableOp")sequential/dense_2/BiasAdd/ReadVariableOp(1�������?9�������?A�������?I�������?a+y��?!?iq�K���?�Unknown
�HHostCast"?sparse_categorical_crossentropy/weighted_loss/num_elements/Cast(1�������?9�������?A�������?I�������?a+y��?!?i�l&	��?�Unknown
vIHostAssignAddVariableOp"AssignAddVariableOp_1(1ffffff�?9ffffff�?Affffff�?Iffffff�?a"��/?i��%����?�Unknown
yJHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1333333�?9333333�?A333333�?I333333�?a�5���?i��I����?�Unknown
aKHostIdentity"Identity(1      �?9      �?A      �?I      �?avW�?ig�� v��?�Unknown�
uLHostReadVariableOp"div_no_nan/ReadVariableOp(1�������?9�������?A�������?I�������?a+y��??i     �?�Unknown*�H
uHostFlushSummaryWriter"FlushSummaryWriter(1������@9������@A������@I������@a��V��?i��V��?�Unknown�
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1�����c@9�����c@A�����c@I�����c@au�Ic�?iVkT�a�?�Unknown
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1�����\b@9�����\b@A�����\b@I�����\b@a����"�?i�c�,��?�Unknown
qHost_FusedMatMul"sequential/dense_1/Relu(1fffffFW@9fffffFW@AfffffFW@IfffffFW@ah������?i;3�����?�Unknown
dHostDataset"Iterator::Model(1�����S@9�����S@A�����,Q@I�����,Q@a�E���f�?ii����?�Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(1�����Q@9�����Q@A�����Q@I�����Q@aPTfP;�?i�ϑ�d�?�Unknown
oHost_FusedMatMul"sequential/dense/Relu(133333�L@933333�L@A33333�L@I33333�L@aq ���?i���x��?�Unknown
oHostSoftmax"sequential/dense_2/Softmax(1      C@9      C@A      C@I      C@a�-Db��?i`�~-j�?�Unknown
�	HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1������7@9������7@A������7@I������7@aM���?i�X{�}��?�Unknown
t
Host_FusedMatMul"sequential/dense_2/BiasAdd(1������6@9������6@A������6@I������6@aTG>�~?i�~	��?�Unknown
^HostGatherV2"GatherV2(1�����L6@9�����L6@A�����L6@I�����L6@a��4�b~?i��qK�$�?�Unknown
{HostMatMul"'gradient_tape/sequential/dense_2/MatMul(1������5@9������5@A������5@I������5@a�!�0�}?i�L�A`�?�Unknown
}HostMatMul")gradient_tape/sequential/dense_2/MatMul_1(1ffffff1@9ffffff1@Affffff1@Iffffff1@a�4�Qf�w?i>�y���?�Unknown
�Host#SparseSoftmaxCrossEntropyWithLogits"gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits(1������.@9������.@A������.@I������.@ar=u}��t?i�����?�Unknown
}HostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1ffffff-@9ffffff-@Affffff-@Iffffff-@ax M��t?i���3���?�Unknown
�HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1ffffff.@9ffffff.@Affffff(@Iffffff(@aFK�\��p?iQ�P��?�Unknown
�HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1      /@9      /@A333333'@I333333'@a��5m��o?i�Qَ"�?�Unknown
iHostWriteSummary"WriteSummary(1ffffff%@9ffffff%@Affffff%@Iffffff%@a�#���(m?i ���?�?�Unknown�
�HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1������@9������@A������@I������@a�;�AK�e?iM�=�>U�?�Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1������@9������@A������@I������@ar=u}��d?i�!��:j�?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1333333@9333333@A333333@I333333@a9��b?iO/�x�|�?�Unknown
HostReluGrad")gradient_tape/sequential/dense_1/ReluGrad(1      @9      @A      @I      @a�G�䛶a?i��y��?�Unknown
�HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1������@9������@A������@I������@a��߂�pa?i`�-���?�Unknown
�HostMul"Ugradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul(1333333@9333333@A333333@I333333@a��5m��_?i.�d1���?�Unknown
�HostSum"1sparse_categorical_crossentropy/weighted_loss/Sum(1333333@9333333@A333333@I333333@a��5m��_?i�/�u���?�Unknown
`HostGatherV2"
GatherV2_1(1������@9������@A������@I������@a���\mn]?i͜I�=��?�Unknown
ZHostArgMax"ArgMax(1333333@9333333@A333333@I333333@a`���\?i�����?�Unknown
gHostStridedSlice"strided_slice(1������@9������@A������@I������@a���_W\?i�S�����?�Unknown
�HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1333333@9333333@A333333@I333333@a�O�D)Z?i�{�q���?�Unknown
lHostIteratorGetNext"IteratorGetNext(1������@9������@A������@I������@a��8 ��Y?i ��P��?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1333333@9333333@A333333@I333333@a����oW?i{�Z"v�?�Unknown
� HostPack"/sparse_categorical_crossentropy/Reshape_1/shape(1333333@9333333@A333333@I333333@a����oW?i�t��-�?�Unknown
�!HostBiasAddGrad"4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGrad(1ffffff@9ffffff@Affffff@Iffffff@a��g�XV?i3L�>Z'�?�Unknown
�"HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1333333@9333333@A333333@I333333@a���߇AU?i�v�1�?�Unknown
�#HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1333333@9333333@A333333@I333333@a���߇AU?i��eƛ<�?�Unknown
x$HostDataset"#Iterator::Model::ParallelMapV2::Zip(1������A@9������A@A������@I������@a
�RWz*T?iOv��F�?�Unknown
|%HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1333333@9333333@A333333@I333333@a9��R?i2����O�?�Unknown
�&HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1      @9      �?A      @I      �?a#̱��YP?iV�!X�?�Unknown
X'HostCast"Cast_1(1333333@9333333@A333333@I333333@a��5m��O?i�/�`�?�Unknown
�(HostTile"Bgradient_tape/sparse_categorical_crossentropy/weighted_loss/Tile_1(1333333@9333333@A333333@I333333@a��5m��O?i��J �g�?�Unknown
e)Host
LogicalAnd"
LogicalAnd(1ffffff@9ffffff@Affffff@Iffffff@a0��z�N?i�2�o�?�Unknown�
�*HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1ffffff@9ffffff@Affffff@Iffffff@a0��z�N?i�t��2w�?�Unknown
V+HostCast"Cast(1������@9������@A������@I������@a���\mnM?i�y�~�?�Unknown
b,HostDivNoNan"div_no_nan_1(1������@9������@A������@I������@a���\mnM?i��k��?�Unknown
�-HostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1������@9������@A������@I������@a���_WL?ira����?�Unknown
�.HostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1������@9������@A������@I������@a���_WL?i\7V���?�Unknown
�/HostCast"bsparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast_1(1������@9������@A������@I������@a���_WL?iFbK�+��?�Unknown
X0HostEqual"Equal(1      @9      @A      @I      @a��}LR@K?i��ް���?�Unknown
t1HostAssignAddVariableOp"AssignAddVariableOp(1ffffff@9ffffff@Affffff@Iffffff@az�!<7I?i��>@��?�Unknown
�2HostReadVariableOp"(sequential/dense_2/MatMul/ReadVariableOp(1ffffff@9ffffff@Affffff@Iffffff@az�!<7I?i��|̄��?�Unknown
v3HostAssignAddVariableOp"AssignAddVariableOp_4(1������@9������@A������@I������@a��)�G?iu�閃��?�Unknown
V4HostSum"Sum_2(1������@9������@A������@I������@a��)�G?ib�Va���?�Unknown
X5HostCast"Cast_3(1������ @9������ @A������ @I������ @ae��+�F?i�}ah;��?�Unknown
�6HostStridedSlice"-sparse_categorical_crossentropy/strided_slice(1������ @9������ @A������ @I������ @ae��+�F?i>olo���?�Unknown
�7HostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1�������?9�������?A�������?I�������?a��;��C?i.>Q,���?�Unknown
w8HostReadVariableOp"div_no_nan_1/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?a9��B?i��%~��?�Unknown
X9HostCast"Cast_2(1      �?9      �?A      �?I      �?a#̱��Y@?i������?�Unknown
�:HostDivNoNan"Lgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/div_no_nan(1      �?9      �?A      �?I      �?a#̱��Y@?i�ZQ���?�Unknown
v;HostAssignAddVariableOp"AssignAddVariableOp_2(1�������?9�������?A�������?I�������?a���_W<?i��K�5��?�Unknown
�<HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1������0@9������0@A�������?I�������?a���_W<?io�F����?�Unknown
T=HostMul"Mul(1333333�?9333333�?A333333�?I333333�?a�O�D):?ie���?�Unknown
s>HostReadVariableOp"SGD/Cast/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?a�O�D):?i[�w4K��?�Unknown
`?HostDivNoNan"
div_no_nan(1333333�?9333333�?A333333�?I333333�?a�O�D):?iQ#]���?�Unknown
X@HostCast"Cast_4(1�������?9�������?A�������?I�������?a��)�7?iǡF��?�Unknown
�AHostDivNoNan"3sparse_categorical_crossentropy/weighted_loss/value(1�������?9�������?A�������?I�������?a��)�7?i= }'���?�Unknown
vBHostAssignAddVariableOp"AssignAddVariableOp_3(1�������?9�������?A�������?I�������?a��;��3?i�����?�Unknown
�CHostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1�������?9�������?A�������?I�������?a��;��3?i-�a�v��?�Unknown
wDHostReadVariableOp"div_no_nan/ReadVariableOp_1(1�������?9�������?A�������?I�������?a��߂�p1?i&Kr����?�Unknown
�EHostReadVariableOp")sequential/dense_2/BiasAdd/ReadVariableOp(1�������?9�������?A�������?I�������?a��߂�p1?i�����?�Unknown
�FHostCast"?sparse_categorical_crossentropy/weighted_loss/num_elements/Cast(1�������?9�������?A�������?I�������?a��߂�p1?i�5��?�Unknown
vGHostAssignAddVariableOp"AssignAddVariableOp_1(1ffffff�?9ffffff�?Affffff�?Iffffff�?a0��z�.?i�SA����?�Unknown
yHHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1333333�?9333333�?A333333�?I333333�?a�O�D)*?i���!���?�Unknown
aIHostIdentity"Identity(1      �?9      �?A      �?I      �?aں���%?i	�w����?�Unknown�
uJHostReadVariableOp"div_no_nan/ReadVariableOp(1�������?9�������?A�������?I�������?a��߂�p!?i     �?�Unknown