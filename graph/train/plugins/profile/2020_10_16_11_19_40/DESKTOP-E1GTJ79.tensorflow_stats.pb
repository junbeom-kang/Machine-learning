"�I
DDeviceIDLE"IDLE1�����ƠBA�����ƠBQ      �?Y      �?�Unknown
BHostIDLE"IDLE1�������@A�������@a>�qQ:h�?i>�qQ:h�?�Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1������@9������@A������@I������@a� Bg?�?i��ٮ���?�Unknown�
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(133333�i@933333�i@A33333�i@I33333�i@a#|<��š?i����)��?�Unknown
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1fffff&h@9fffff&h@Afffff&h@Ifffff&h@a���7��?i�8�@m��?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1fffffX@9fffffX@AfffffX@IfffffX@a���M?~�?i��:_}�?�Unknown
qHost_FusedMatMul"sequential/dense_1/Relu(1fffff&R@9fffff&R@Afffff&R@Ifffff&R@a�r2��?i�I�>��?�Unknown
oHost_FusedMatMul"sequential/dense/Relu(1�����Q@9�����Q@A�����Q@I�����Q@a����h�?i�s���>�?�Unknown
d	HostDataset"Iterator::Model(1fffffT@9fffffT@A33333�K@I33333�K@a~�+�?ip�Y#���?�Unknown
y
HostMatMul"%gradient_tape/sequential/dense/MatMul(1ffffffE@9ffffffE@AffffffE@IffffffE@a)C+��a}?i��Є��?�Unknown
oHostSoftmax"sequential/dense_2/Softmax(1������B@9������B@A������B@I������B@a�����y?i2S�G��?�Unknown
^HostGatherV2"GatherV2(1������<@9������<@A������<@I������<@aw�Ħl�s?i_�`�� �?�Unknown
tHost_FusedMatMul"sequential/dense_2/BiasAdd(1ffffff;@9ffffff;@Affffff;@Iffffff;@a��S	b�r?i6�s�qF�?�Unknown
�Host#SparseSoftmaxCrossEntropyWithLogits"gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits(1ffffff;@9ffffff;@Affffff;@Iffffff;@a��S	b�r?i,�Il�?�Unknown
sHostDataset"Iterator::Model::ParallelMapV2(133333�8@933333�8@A33333�8@I33333�8@a}}����p?i�9	���?�Unknown
�HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      ;@9      ;@A3333338@I3333338@aR�v� �p?i�||
4��?�Unknown
�HostStridedSlice"-sparse_categorical_crossentropy/strided_slice(1     �5@9     �5@A     �5@I     �5@an����m?i	&`���?�Unknown
�HostTile"Bgradient_tape/sparse_categorical_crossentropy/weighted_loss/Tile_1(1      5@9      5@A      5@I      5@a4s>�l?iZ�E���?�Unknown
{HostMatMul"'gradient_tape/sequential/dense_2/MatMul(1      3@9      3@A      3@I      3@a��_�Dj?i乄���?�Unknown
}HostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1�����0@9�����0@A�����0@I�����0@aZ%�f?i>٩~��?�Unknown
}HostMatMul")gradient_tape/sequential/dense_2/MatMul_1(1������,@9������,@A������,@I������,@a3�F�F�c?i :�a-�?�Unknown
�HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      *@9      *@A      *@I      *@a�@�kW�a?iF�;?�?�Unknown
�HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1ffffff.@9ffffff.@A������)@I������)@a1��>�a?i���'�P�?�Unknown
iHostWriteSummary"WriteSummary(1ffffff!@9ffffff!@Affffff!@Iffffff!@az��I��W?iX���\�?�Unknown�
�HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1333333@9333333@A333333@I333333@a���LQ?iZ�yfe�?�Unknown
HostReluGrad")gradient_tape/sequential/dense_1/ReluGrad(1333333@9333333@A333333@I333333@a��b��O?iO#+]m�?�Unknown
�HostCast"bsparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast_1(1������@9������@A������@I������@a��MO?iu&-�0u�?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1ffffff@9ffffff@Affffff@Iffffff@a%�l*`CI?i��7m�{�?�Unknown
�HostMul"Ugradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul(1ffffff@9ffffff@Affffff@Iffffff@a%�l*`CI?i�\BEҁ�?�Unknown
lHostIteratorGetNext"IteratorGetNext(1      @9      @A      @I      @a�u�ǶH?i`z6����?�Unknown
`HostGatherV2"
GatherV2_1(1ffffff@9ffffff@Affffff@Iffffff@a�ǘhf�F?i������?�Unknown
� HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1ffffff@9ffffff@Affffff@Iffffff@a�ǘhf�F?i��j*B��?�Unknown
�!HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      @9      @A      @I      @a�����E?i'o����?�Unknown
Z"HostArgMax"ArgMax(1ffffff@9ffffff@Affffff@Iffffff@a��Z��D?i�E����?�Unknown
x#HostDataset"#Iterator::Model::ParallelMapV2::Zip(1fffff�G@9fffff�G@Affffff@Iffffff@a��Z��D?i�țl/��?�Unknown
�$HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1ffffff@9ffffff@Affffff@Iffffff@a��Z��D?isu�g��?�Unknown
�%HostSum"1sparse_categorical_crossentropy/weighted_loss/Sum(1������@9������@A������@I������@a�ӻ RD?ih�2�{��?�Unknown
g&HostStridedSlice"strided_slice(1������@9������@A������@I������@a�ӻ RD?i]�r���?�Unknown
�'HostBiasAddGrad"4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGrad(1333333@9333333@A333333@I333333@aS��;�B?i�o%;��?�Unknown
�(HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1������	@9������	@A������	@I������	@a1��>�A?i�B?蟻�?�Unknown
�)HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1ffffff@9ffffff�?Affffff@Iffffff�?a�[�S�>?ix�x��?�Unknown
V*HostSum"Sum_2(1������@9������@A������@I������@a��&�"�=?iTJ-��?�Unknown
�+HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1333333@9333333@A333333@I333333@aIs[ސ\:?i�0�x��?�Unknown
X,HostCast"Cast_1(1������@9������@A������@I������@a~v/*8?i���}��?�Unknown
X-HostCast"Cast_4(1������@9������@A������@I������@a~v/*8?ib�5���?�Unknown
e.Host
LogicalAnd"
LogicalAnd(1������@9������@A������@I������@a~v/*8?i2��z���?�Unknown�
�/HostPack"/sparse_categorical_crossentropy/Reshape_1/shape(1ffffff�?9ffffff�?Affffff�?Iffffff�?a��Z��4?i�ۧN$��?�Unknown
`0HostDivNoNan"
div_no_nan(1�������?9�������?A�������?I�������?aw�Ħl�3?i'�<����?�Unknown
b1HostDivNoNan"div_no_nan_1(1�������?9�������?A�������?I�������?aw�Ħl�3?i��ѩ��?�Unknown
t2HostAssignAddVariableOp"AssignAddVariableOp(1333333�?9333333�?A333333�?I333333�?aS��;�2?i~�O1k��?�Unknown
|3HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1333333�?9333333�?A333333�?I333333�?aS��;�2?iBBθ���?�Unknown
u4HostReadVariableOp"div_no_nan/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?aS��;�2?i�L@��?�Unknown
�5HostReadVariableOp")sequential/dense_2/BiasAdd/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?aS��;�2?i����k��?�Unknown
�6HostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1�������?9�������?A�������?I�������?a1��>�1?i��2)���?�Unknown
�7HostReadVariableOp"(sequential/dense_2/MatMul/ReadVariableOp(1      �?9      �?A      �?I      �?a(���y0?i�3�d���?�Unknown
v8HostAssignAddVariableOp"AssignAddVariableOp_2(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�[�S�.?i9�y���?�Unknown
V9HostCast"Cast(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�[�S�.?i�������?�Unknown
�:HostDivNoNan"3sparse_categorical_crossentropy/weighted_loss/value(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�[�S�.?i��4�q��?�Unknown
v;HostAssignAddVariableOp"AssignAddVariableOp_4(1�������?9�������?A�������?I�������?a�g8F�,?ik;Y�:��?�Unknown
�<HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1     �0@9     �0@A�������?I�������?a�g8F�,?i�}���?�Unknown
T=HostMul"Mul(1333333�?9333333�?A333333�?I333333�?aIs[ސ\*?i���K���?�Unknown
�>HostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?aIs[ސ\*?i_j�O��?�Unknown
X?HostEqual"Equal(1�������?9�������?A�������?I�������?a~v/*(?iGҐ����?�Unknown
X@HostCast"Cast_2(1      �?9      �?A      �?I      �?a�����%?i`�q41��?�Unknown
�AHostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1      �?9      �?A      �?I      �?a�����%?iy�R����?�Unknown
vBHostAssignAddVariableOp"AssignAddVariableOp_1(1�������?9�������?A�������?I�������?aw�Ħl�#?i����?�Unknown
sCHostReadVariableOp"SGD/Cast/ReadVariableOp(1�������?9�������?A�������?I�������?aw�Ħl�#?i�^	��?�Unknown
�DHostDivNoNan"Lgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/div_no_nan(1�������?9�������?A�������?I�������?aw�Ħl�#?iT뱵E��?�Unknown
XEHostCast"Cast_3(1�������?9�������?A�������?I�������?a1��>�!?i��e�^��?�Unknown
wFHostReadVariableOp"div_no_nan_1/ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�[�S�?iyJ�T��?�Unknown
�GHostCast"?sparse_categorical_crossentropy/weighted_loss/num_elements/Cast(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�[�S�?i$���J��?�Unknown
wHHostReadVariableOp"div_no_nan/ReadVariableOp_1(1333333�?9333333�?A333333�?I333333�?aIs[ސ\?i �'���?�Unknown
yIHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1333333�?9333333�?A333333�?I333333�?aIs[ސ\?iܠ�����?�Unknown
vJHostAssignAddVariableOp"AssignAddVariableOp_3(1      �?9      �?A      �?I      �?a�����?i�����?�Unknown
aKHostIdentity"Identity(1      �?9      �?A      �?I      �?a�����?i�AP��?�Unknown�
�LHostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1      �?9      �?A      �?I      �?a�����?i      �?�Unknown*�H
uHostFlushSummaryWriter"FlushSummaryWriter(1������@9������@A������@I������@a�����?i�����?�Unknown�
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(133333�i@933333�i@A33333�i@I33333�i@a��2����?i��J���?�Unknown
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1fffff&h@9fffff&h@Afffff&h@Ifffff&h@ad�Y��ϭ?i3������?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1fffffX@9fffffX@AfffffX@IfffffX@a�^��?i¢t7*��?�Unknown
qHost_FusedMatMul"sequential/dense_1/Relu(1fffff&R@9fffff&R@Afffff&R@Ifffff&R@aX�z �g�?iUzx�fc�?�Unknown
oHost_FusedMatMul"sequential/dense/Relu(1�����Q@9�����Q@A�����Q@I�����Q@a�gl��?i�-����?�Unknown
dHostDataset"Iterator::Model(1fffffT@9fffffT@A33333�K@I33333�K@a�}��?i �����?�Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(1ffffffE@9ffffffE@AffffffE@IffffffE@a_/^X�j�?i���5��?�Unknown
o	HostSoftmax"sequential/dense_2/Softmax(1������B@9������B@A������B@I������B@a�a���D�?iEİ�H[�?�Unknown
^
HostGatherV2"GatherV2(1������<@9������<@A������<@I������<@aQӯ�Ɓ?i���b��?�Unknown
tHost_FusedMatMul"sequential/dense_2/BiasAdd(1ffffff;@9ffffff;@Affffff;@Iffffff;@a�c�iM�?i!�%���?�Unknown
�Host#SparseSoftmaxCrossEntropyWithLogits"gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits(1ffffff;@9ffffff;@Affffff;@Iffffff;@a�c�iM�?i�v�3�)�?�Unknown
sHostDataset"Iterator::Model::ParallelMapV2(133333�8@933333�8@A33333�8@I33333�8@a�ϋub}~?iP����f�?�Unknown
�HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1      ;@9      ;@A3333338@I3333338@a����a�}?i,ָ�f��?�Unknown
�HostStridedSlice"-sparse_categorical_crossentropy/strided_slice(1     �5@9     �5@A     �5@I     �5@a,��")�z?i��{��?�Unknown
�HostTile"Bgradient_tape/sparse_categorical_crossentropy/weighted_loss/Tile_1(1      5@9      5@A      5@I      5@a+.(�y?i@�Z^S�?�Unknown
{HostMatMul"'gradient_tape/sequential/dense_2/MatMul(1      3@9      3@A      3@I      3@a'�kZ$tw?iX��;:�?�Unknown
}HostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1�����0@9�����0@A�����0@I�����0@a��g��s?if���a�?�Unknown
}HostMatMul")gradient_tape/sequential/dense_2/MatMul_1(1������,@9������,@A������,@I������,@a�)�q?i1�H��?�Unknown
�HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1      *@9      *@A      *@I      *@a��p?ig�a��?�Unknown
�HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1ffffff.@9ffffff.@A������)@I������)@a�Ɣʙo?i4�����?�Unknown
iHostWriteSummary"WriteSummary(1ffffff!@9ffffff!@Affffff!@Iffffff!@aV��ze?iS6lu��?�Unknown�
�HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1333333@9333333@A333333@I333333@aͱsjc_?i,;���?�Unknown
HostReluGrad")gradient_tape/sequential/dense_1/ReluGrad(1333333@9333333@A333333@I333333@a�)Ԗ_�\?iA���T��?�Unknown
�HostCast"bsparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast_1(1������@9������@A������@I������@a��l�$\?i���Ig�?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1ffffff@9ffffff@Affffff@Iffffff@aY��V?i:]����?�Unknown
�HostMul"Ugradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul(1ffffff@9ffffff@Affffff@Iffffff@aY��V?i�ԇ��?�Unknown
lHostIteratorGetNext"IteratorGetNext(1      @9      @A      @I      @a%țp"8V?i�"��9(�?�Unknown
`HostGatherV2"
GatherV2_1(1ffffff@9ffffff@Affffff@Iffffff@aT[Oǅ>T?i>ʣ'Y2�?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1ffffff@9ffffff@Affffff@Iffffff@aT[Oǅ>T?i�q�jx<�?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1      @9      @A      @I      @a!@���S?i��yXF�?�Unknown
Z HostArgMax"ArgMax(1ffffff@9ffffff@Affffff@Iffffff@a�	VHP�R?i�!�O�?�Unknown
x!HostDataset"#Iterator::Model::ParallelMapV2::Zip(1fffff�G@9fffff�G@Affffff@Iffffff@a�	VHP�R?iF�Y�?�Unknown
�"HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1ffffff@9ffffff@Affffff@Iffffff@a�	VHP�R?iqBr}b�?�Unknown
�#HostSum"1sparse_categorical_crossentropy/weighted_loss/Sum(1������@9������@A������@I������@a���DR?i�r��k�?�Unknown
g$HostStridedSlice"strided_slice(1������@9������@A������@I������@a���DR?i	t`[�t�?�Unknown
�%HostBiasAddGrad"4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGrad(1333333@9333333@A333333@I333333@a�	���P?i��/5'}�?�Unknown
�&HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1������	@9������	@A������	@I������	@a�ƔʙO?i�*է��?�Unknown
�'HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1ffffff@9ffffff�?Affffff@Iffffff�?aa�-B��K?i�%L���?�Unknown
V(HostSum"Sum_2(1������@9������@A������@I������@a����©J?i�ἡ��?�Unknown
�)HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1333333@9333333@A333333@I333333@a���W�G?i<�ܒ���?�Unknown
X*HostCast"Cast_1(1������@9������@A������@I������@a�HF��E?ig�����?�Unknown
X+HostCast"Cast_4(1������@9������@A������@I������@a�HF��E?i�!�pk��?�Unknown
e,Host
LogicalAnd"
LogicalAnd(1������@9������@A������@I������@a�HF��E?i��Q�٨�?�Unknown�
�-HostPack"/sparse_categorical_crossentropy/Reshape_1/shape(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�	VHP�B?i?�c����?�Unknown
`.HostDivNoNan"
div_no_nan(1�������?9�������?A�������?I�������?aQӯ��A?i4��S���?�Unknown
b/HostDivNoNan"div_no_nan_1(1�������?9�������?A�������?I�������?aQӯ��A?i)�]�m��?�Unknown
t0HostAssignAddVariableOp"AssignAddVariableOp(1333333�?9333333�?A333333�?I333333�?a�	���@?i�cEa���?�Unknown
|1HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1333333�?9333333�?A333333�?I333333�?a�	���@?i�%-�Ҿ�?�Unknown
u2HostReadVariableOp"div_no_nan/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?a�	���@?i^�;��?�Unknown
�3HostReadVariableOp")sequential/dense_2/BiasAdd/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?a�	���@?iŪ��7��?�Unknown
�4HostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1�������?9�������?A�������?I�������?a�Ɣʙ??i�CO�*��?�Unknown
�5HostReadVariableOp"(sequential/dense_2/MatMul/ReadVariableOp(1      �?9      �?A      �?I      �?a1`z�-�=?i�����?�Unknown
v6HostAssignAddVariableOp"AssignAddVariableOp_2(1ffffff�?9ffffff�?Affffff�?Iffffff�?aa�-B��;?i��4�S��?�Unknown
V7HostCast"Cast(1ffffff�?9ffffff�?Affffff�?Iffffff�?aa�-B��;?ig>]����?�Unknown
�8HostDivNoNan"3sparse_categorical_crossentropy/weighted_loss/value(1ffffff�?9ffffff�?Affffff�?Iffffff�?aa�-B��;?i%��]=��?�Unknown
v9HostAssignAddVariableOp"AssignAddVariableOp_4(1�������?9�������?A�������?I�������?a�����9?iV��r��?�Unknown
�:HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1     �0@9     �0@A�������?I�������?a�����9?i�������?�Unknown
T;HostMul"Mul(1333333�?9333333�?A333333�?I333333�?a���W�7?i*�����?�Unknown
�<HostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?a���W�7?i͡�p���?�Unknown
X=HostEqual"Equal(1�������?9�������?A�������?I�������?a�HF��5?i�j�L��?�Unknown
X>HostCast"Cast_2(1      �?9      �?A      �?I      �?a!@���3?ik
����?�Unknown
�?HostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1      �?9      �?A      �?I      �?a!@���3?i󩷯<��?�Unknown
v@HostAssignAddVariableOp"AssignAddVariableOp_1(1�������?9�������?A�������?I�������?aQӯ��1?i��u��?�Unknown
sAHostReadVariableOp"SGD/Cast/ReadVariableOp(1�������?9�������?A�������?I�������?aQӯ��1?i�4P���?�Unknown
�BHostDivNoNan"Lgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/div_no_nan(1�������?9�������?A�������?I�������?aQӯ��1?i�s ���?�Unknown
XCHostCast"Cast_3(1�������?9�������?A�������?I�������?a�Ɣʙ/?iNX����?�Unknown
wDHostReadVariableOp"div_no_nan_1/ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?aa�-B��+?i-{0&���?�Unknown
�EHostCast"?sparse_categorical_crossentropy/weighted_loss/num_elements/Cast(1ffffff�?9ffffff�?Affffff�?Iffffff�?aa�-B��+?i�D�U��?�Unknown
wFHostReadVariableOp"div_no_nan/ReadVariableOp_1(1333333�?9333333�?A333333�?I333333�?a���W�'?i^������?�Unknown
yGHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1333333�?9333333�?A333333�?I333333�?a���W�'?i��B�K��?�Unknown
vHHostAssignAddVariableOp"AssignAddVariableOp_3(1      �?9      �?A      �?I      �?a!@���#?it`,����?�Unknown
aIHostIdentity"Identity(1      �?9      �?A      �?I      �?a!@���#?i80����?�Unknown�
�JHostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1      �?9      �?A      �?I      �?a!@���#?i�������?�Unknown