"�K
DDeviceIDLE"IDLE1�����ƠBA�����ƠBQ      �?Y      �?�Unknown
BHostIDLE"IDLE1����Lĳ@A����Lĳ@a��`zp��?i��`zp��?�Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1     I�@9     I�@A     I�@I     I�@a��$���?iqtsdi��?�Unknown�
}HostMatMul")gradient_tape/sequential_1/dense_3/MatMul(1fffff�p@9fffff�p@Afffff�p@Ifffff�p@a���@���?iֈz�E��?�Unknown
sHost_FusedMatMul"sequential_1/dense_3/Relu(1�����ip@9�����ip@A�����ip@I�����ip@a�Ou�
u�?iT3�T��?�Unknown
�HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1�����yX@9�����yX@A�����yX@I�����yX@ah]�׃t�?iʸd��?�Unknown
^HostGatherV2"GatherV2(1����̌O@9����̌O@A����̌O@I����̌O@a��78><~?io(}�8H�?�Unknown
}HostMatMul")gradient_tape/sequential_1/dense_4/MatMul(133333�N@933333�N@A33333�N@I33333�N@a�)dٵk}?i��/L��?�Unknown
s	Host_FusedMatMul"sequential_1/dense_4/Relu(1333333F@9333333F@A333333F@I333333F@a,�%bFu?i�/z���?�Unknown

HostMatMul"+gradient_tape/sequential_1/dense_4/MatMul_1(1      D@9      D@A      D@I      D@a�Vզ*s?i�$^���?�Unknown
iHostWriteSummary"WriteSummary(133333�3@933333�3@A33333�3@I33333�3@a��tJ�b?i�Pok���?�Unknown�
�Host#SparseSoftmaxCrossEntropyWithLogits"gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits(1ffffff/@9ffffff/@Affffff/@Iffffff/@aO-�rq^?i��($���?�Unknown
�HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(13333332@93333332@A      /@I      /@aztEdO�]?i:��˹�?�Unknown
�HostReluGrad"+gradient_tape/sequential_1/dense_3/ReluGrad(1333333+@9333333+@A333333+@I333333+@a�u�Z?i��Q��?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1������*@9������*@A������*@I������*@a�_���Y?iv�ƙ�?�Unknown
gHostStridedSlice"strided_slice(1������*@9������*@A������*@I������*@aT����}Y?i�d�X+�?�Unknown
HostMatMul"+gradient_tape/sequential_1/dense_5/MatMul_1(1333333*@9333333*@A333333*@I333333*@a~�0��Y?i}k��7�?�Unknown
`HostGatherV2"
GatherV2_1(1      )@9      )@A      )@I      )@a����P�W?i�Ұ6�C�?�Unknown
}HostMatMul")gradient_tape/sequential_1/dense_5/MatMul(1ffffff(@9ffffff(@Affffff(@Iffffff(@a�
�tbW?irGkE�O�?�Unknown
dHostDataset"Iterator::Model(1      1@9      1@A������'@I������'@a�u&_��V?i�ښ��Z�?�Unknown
qHostSoftmax"sequential_1/dense_5/Softmax(1333333'@9333333'@A333333'@I333333'@aA�cI�;V?i��?�f�?�Unknown
xHostDataset"#Iterator::Model::ParallelMapV2::Zip(1fffff�C@9fffff�C@A      !@I      !@a\/�h�JP?i5���<n�?�Unknown
�HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1������&@9������&@A333333@I333333@a�P�k`�M?iɲ��u�?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1ffffff@9ffffff@Affffff@Iffffff@a:߂N"M?i�S"	�|�?�Unknown
�HostMul"Ugradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul(1      @9      @A      @I      @ae&@��L?i�S�/��?�Unknown
eHost
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @a�Qgf��F?i����?�Unknown�
�HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @a�Qgf��F?is�e���?�Unknown
�HostReluGrad"+gradient_tape/sequential_1/dense_4/ReluGrad(1������@9������@A������@I������@a��WٝF?iـ�{V��?�Unknown
lHostIteratorGetNext"IteratorGetNext(1ffffff@9ffffff@Affffff@Iffffff@a�n`,swE?i���X���?�Unknown
vHost_FusedMatMul"sequential_1/dense_5/BiasAdd(1������@9������@A������@I������@a��\/�D?i4pJ$��?�Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1ffffff@9ffffff@Affffff@Iffffff@an���ȌC?i)f�VĤ�?�Unknown
X HostCast"Cast_1(1ffffff@9ffffff@Affffff@Iffffff@aE6O��A?i�9*�,��?�Unknown
Z!HostArgMax"ArgMax(1333333@9333333@A333333@I333333@a��o�{@?iz,F�K��?�Unknown
�"HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1������@9������@A������@I������@a�RHa�@?i�~�1R��?�Unknown
V#HostSum"Sum_2(1      @9      @A      @I      @a����>?iǏo�'��?�Unknown
�$HostCast"bsparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast_1(1ffffff@9ffffff@Affffff@Iffffff@a:߂N"=?i#`��˸�?�Unknown
�%HostSum"1sparse_categorical_crossentropy/weighted_loss/Sum(1������@9������@A������@I������@a�m1�]<?i���W��?�Unknown
�&HostTile"Bgradient_tape/sparse_categorical_crossentropy/weighted_loss/Tile_1(1������@9������@A������@I������@a��{��;?i���ʿ�?�Unknown
�'HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @a=�x�O�:?i��a%��?�Unknown
�(HostStridedSlice"-sparse_categorical_crossentropy/strided_slice(1      @9      @A      @I      @a=�x�O�:?i�����?�Unknown
�)HostBiasAddGrad"6gradient_tape/sequential_1/dense_3/BiasAdd/BiasAddGrad(1333333@9333333@A333333@I333333@a�u�:?iUL�,���?�Unknown
�*HostBiasAddGrad"6gradient_tape/sequential_1/dense_4/BiasAdd/BiasAddGrad(1333333@9333333@A333333@I333333@a�u�:?i��vN��?�Unknown
�+HostBiasAddGrad"6gradient_tape/sequential_1/dense_5/BiasAdd/BiasAddGrad(1������	@9������	@A������	@I������	@a?5n���8?i���^��?�Unknown
�,HostCast"BArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_Cast_1(1      @9      @A      @I      @a�Qgf��6?i�uW^���?�Unknown
�-HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @a�Qgf��6?i�B�]���?�Unknown
V.HostCast"Cast(1ffffff@9ffffff@Affffff@Iffffff@a�n`,sw5?i��)L���?�Unknown
�/HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1������@9�������?A������@I�������?a��\/�4?iA����?�Unknown
|0HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1      @9      @A      @I      @a�Vզ*3?ie����?�Unknown
X1HostCast"Cast_4(1������@9������@A������@I������@a��K~��0?i}.6��?�Unknown
�2HostPack"/sparse_categorical_crossentropy/Reshape_1/shape(1������@9������@A������@I������@a��K~��0?i���}���?�Unknown
v3HostAssignAddVariableOp"AssignAddVariableOp_2(1������ @9������ @A������ @I������ @a�RHa�0?i !R����?�Unknown
�4HostReadVariableOp"*sequential_1/dense_5/MatMul/ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?a:߂N"-?i.	Ҍ��?�Unknown
t5HostAssignAddVariableOp"AssignAddVariableOp(1�������?9�������?A�������?I�������?a��{��+?i�PXkF��?�Unknown
X6HostEqual"Equal(1333333�?9333333�?A333333�?I333333�?a�u�*?i@�|���?�Unknown
�7HostDivNoNan"Lgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/div_no_nan(1�������?9�������?A�������?I�������?a?5n���(?i#�Op��?�Unknown
v8HostAssignAddVariableOp"AssignAddVariableOp_1(1      �?9      �?A      �?I      �?a�Qgf��&?i�e���?�Unknown
�9HostReadVariableOp"+sequential_1/dense_3/BiasAdd/ReadVariableOp(1      �?9      �?A      �?I      �?a�Qgf��&?i̼P��?�Unknown
�:HostCast"?sparse_categorical_crossentropy/weighted_loss/num_elements/Cast(1      �?9      �?A      �?I      �?a�Qgf��&?i�2s���?�Unknown
T;HostMul"Mul(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�n`,sw%?i���z��?�Unknown
�<HostReadVariableOp"*sequential_1/dense_4/MatMul/ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�n`,sw%?i����n��?�Unknown
�=HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1333333)@9333333)@A�������?I�������?aC�Y���#?i)����?�Unknown
s>HostReadVariableOp"SGD/Cast/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?a�R�bf"?iSi�F���?�Unknown
b?HostDivNoNan"div_no_nan_1(1333333�?9333333�?A333333�?I333333�?a�R�bf"?i}�ެ���?�Unknown
v@HostAssignAddVariableOp"AssignAddVariableOp_4(1�������?9�������?A�������?I�������?a��K~�� ?i9ӆ���?�Unknown
`AHostDivNoNan"
div_no_nan(1�������?9�������?A�������?I�������?a��K~�� ?i��.h��?�Unknown
XBHostCast"Cast_2(1      �?9      �?A      �?I      �?a����?iC�R���?�Unknown
wCHostReadVariableOp"div_no_nan_1/ReadVariableOp(1�������?9�������?A�������?I�������?a��{��?i#�����?�Unknown
�DHostReadVariableOp"+sequential_1/dense_5/BiasAdd/ReadVariableOp(1�������?9�������?A�������?I�������?a��{��?iD�V���?�Unknown
�EHostDivNoNan"3sparse_categorical_crossentropy/weighted_loss/value(1�������?9�������?A�������?I�������?a��{��?i��4#���?�Unknown
vFHostAssignAddVariableOp"AssignAddVariableOp_3(1�������?9�������?A�������?I�������?a?5n���?iU�Qgf��?�Unknown
XGHostCast"Cast_3(1�������?9�������?A�������?I�������?a?5n���?i��n�*��?�Unknown
aHHostIdentity"Identity(1�������?9�������?A�������?I�������?a?5n���?i9�����?�Unknown�
wIHostReadVariableOp"div_no_nan/ReadVariableOp_1(1�������?9�������?A�������?I�������?a?5n���?i���3���?�Unknown
�JHostReadVariableOp"*sequential_1/dense_3/MatMul/ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�n`,sw?i�XB�^��?�Unknown
uKHostReadVariableOp"div_no_nan/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?a�R�bf?iCX"���?�Unknown
�LHostReadVariableOp"+sequential_1/dense_4/BiasAdd/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?a�R�bf?i��mU���?�Unknown
yMHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      �?9      �?A      �?I      �?a����?i�������?�Unknown*�J
uHostFlushSummaryWriter"FlushSummaryWriter(1     I�@9     I�@A     I�@I     I�@aN�G�y�?iN�G�y�?�Unknown�
}HostMatMul")gradient_tape/sequential_1/dense_3/MatMul(1fffff�p@9fffff�p@Afffff�p@Ifffff�p@a�o��Xq�?i@Ż�G��?�Unknown
sHost_FusedMatMul"sequential_1/dense_3/Relu(1�����ip@9�����ip@A�����ip@I�����ip@a�e<e!F�?i�Qc�P�?�Unknown
�HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1�����yX@9�����yX@A�����yX@I�����yX@a1�u����?iw���5�?�Unknown
^HostGatherV2"GatherV2(1����̌O@9����̌O@A����̌O@I����̌O@ab-z��?i�fW�/��?�Unknown
}HostMatMul")gradient_tape/sequential_1/dense_4/MatMul(133333�N@933333�N@A33333�N@I33333�N@a^l�\��?i��>/eZ�?�Unknown
sHost_FusedMatMul"sequential_1/dense_4/Relu(1333333F@9333333F@A333333F@I333333F@axcw
�?i'^Y���?�Unknown
HostMatMul"+gradient_tape/sequential_1/dense_4/MatMul_1(1      D@9      D@A      D@I      D@a��
B�|�?im�$� �?�Unknown
i	HostWriteSummary"WriteSummary(133333�3@933333�3@A33333�3@I33333�3@aE�!�u"w?i�̂��N�?�Unknown�
�
Host#SparseSoftmaxCrossEntropyWithLogits"gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits(1ffffff/@9ffffff/@Affffff/@Iffffff/@a��R�or?iIr���s�?�Unknown
�HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(13333332@93333332@A      /@I      /@a�a��3r?iA6�T,��?�Unknown
�HostReluGrad"+gradient_tape/sequential_1/dense_3/ReluGrad(1333333+@9333333+@A333333+@I333333+@a���4�o?i���?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1������*@9������*@A������*@I������*@a�AD�xo?iD ,|���?�Unknown
gHostStridedSlice"strided_slice(1������*@9������*@A������*@I������*@aB���<o?i�3O���?�Unknown
HostMatMul"+gradient_tape/sequential_1/dense_5/MatMul_1(1333333*@9333333*@A333333*@I333333*@a/2��n?i�e,��?�Unknown
`HostGatherV2"
GatherV2_1(1      )@9      )@A      )@I      )@a�����[m?iO����2�?�Unknown
}HostMatMul")gradient_tape/sequential_1/dense_5/MatMul(1ffffff(@9ffffff(@Affffff(@Iffffff(@a\G��n�l?i����O�?�Unknown
dHostDataset"Iterator::Model(1      1@9      1@A������'@I������'@a���k?i��2-�k�?�Unknown
qHostSoftmax"sequential_1/dense_5/Softmax(1333333'@9333333'@A333333'@I333333'@a&�a�>k?ii���̆�?�Unknown
xHostDataset"#Iterator::Model::ParallelMapV2::Zip(1fffff�C@9fffff�C@A      !@I      !@a�����c?i��Ú�?�Unknown
�HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1������&@9������&@A333333@I333333@a�[Z��Qb?i^��w��?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1ffffff@9ffffff@Affffff@Iffffff@a��x��a?i;q��?�Unknown
�HostMul"Ugradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul(1      @9      @A      @I      @a/��|�a?iX������?�Unknown
eHost
LogicalAnd"
LogicalAnd(1      @9      @A      @I      @aK���-/\?i<f�(���?�Unknown�
�HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1      @9      @A      @I      @aK���-/\?i ӝ����?�Unknown
�HostReluGrad"+gradient_tape/sequential_1/dense_4/ReluGrad(1������@9������@A������@I������@a9I�$��[?iEO06���?�Unknown
lHostIteratorGetNext"IteratorGetNext(1ffffff@9ffffff@Affffff@Iffffff@a�S�*NZ?i+��K��?�Unknown
vHost_FusedMatMul"sequential_1/dense_5/BiasAdd(1������@9������@A������@I������@a�͐Q�]Y?i��E m�?�Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1ffffff@9ffffff@Affffff@Iffffff@a�P���W?i��ȓg �?�Unknown
XHostCast"Cast_1(1ffffff@9ffffff@Affffff@Iffffff@aJՄ2��U?i%�ae5+�?�Unknown
ZHostArgMax"ArgMax(1333333@9333333@A333333@I333333@aX���2T?iQj��N5�?�Unknown
� HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1������@9������@A������@I������@a��"��S?i���%,?�?�Unknown
V!HostSum"Sum_2(1      @9      @A      @I      @a��;��R?i��45�H�?�Unknown
�"HostCast"bsparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast_1(1ffffff@9ffffff@Affffff@Iffffff@a��x��Q?iD�~Q�?�Unknown
�#HostSum"1sparse_categorical_crossentropy/weighted_loss/Sum(1������@9������@A������@I������@a�]�O\aQ?i��.Z�?�Unknown
�$HostTile"Bgradient_tape/sparse_categorical_crossentropy/weighted_loss/Tile_1(1������@9������@A������@I������@a�޵��P?i��?�b�?�Unknown
�%HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1      @9      @A      @I      @a�_���pP?i����j�?�Unknown
�&HostStridedSlice"-sparse_categorical_crossentropy/strided_slice(1      @9      @A      @I      @a�_���pP?i�ss�?�Unknown
�'HostBiasAddGrad"6gradient_tape/sequential_1/dense_3/BiasAdd/BiasAddGrad(1333333@9333333@A333333@I333333@a���4�O?i��ug{�?�Unknown
�(HostBiasAddGrad"6gradient_tape/sequential_1/dense_4/BiasAdd/BiasAddGrad(1333333@9333333@A333333@I333333@a���4�O?i��w���?�Unknown
�)HostBiasAddGrad"6gradient_tape/sequential_1/dense_5/BiasAdd/BiasAddGrad(1������	@9������	@A������	@I������	@a��_�0N?i�ɵ����?�Unknown
�*HostCast"BArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_Cast_1(1      @9      @A      @I      @aK���-/L?iZ 0����?�Unknown
�+HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1      @9      @A      @I      @aK���-/L?i�6�W���?�Unknown
V,HostCast"Cast(1ffffff@9ffffff@Affffff@Iffffff@a�S�*NJ?i��`�;��?�Unknown
�-HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1������@9�������?A������@I�������?a�͐Q�]I?i��L���?�Unknown
|.HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1      @9      @A      @I      @a��
B�|G?i�rEvr��?�Unknown
X/HostCast"Cast_4(1������@9������@A������@I������@a&���!�D?i#�>���?�Unknown
�0HostPack"/sparse_categorical_crossentropy/Reshape_1/shape(1������@9������@A������@I������@a&���!�D?i��ȵ�?�Unknown
v1HostAssignAddVariableOp"AssignAddVariableOp_2(1������ @9������ @A������ @I������ @a��"��C?iH�#����?�Unknown
�2HostReadVariableOp"*sequential_1/dense_5/MatMul/ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?a��x��A?iqh-��?�Unknown
t3HostAssignAddVariableOp"AssignAddVariableOp(1�������?9�������?A�������?I�������?a�޵��@?i�^K]g��?�Unknown
X4HostEqual"Equal(1333333�?9333333�?A333333�?I333333�?a���4�??i�[̃e��?�Unknown
�5HostDivNoNan"Lgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/div_no_nan(1�������?9�������?A�������?I�������?a��_�0>?i�g�'��?�Unknown
v6HostAssignAddVariableOp"AssignAddVariableOp_1(1      �?9      �?A      �?I      �?aK���-/<?iႨo���?�Unknown
�7HostReadVariableOp"+sequential_1/dense_3/BiasAdd/ReadVariableOp(1      �?9      �?A      �?I      �?aK���-/<?i�eU3��?�Unknown
�8HostCast"?sparse_categorical_crossentropy/weighted_loss/num_elements/Cast(1      �?9      �?A      �?I      �?aK���-/<?iS�";���?�Unknown
T9HostMul"Mul(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�S�*N:?i��} ��?�Unknown
�:HostReadVariableOp"*sequential_1/dense_4/MatMul/ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�S�*N:?iG��L��?�Unknown
�;HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1333333)@9333333)@A�������?I�������?a����'m8?iH�jZ��?�Unknown
s<HostReadVariableOp"SGD/Cast/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?ao�G�$�6?i��i�+��?�Unknown
b=HostDivNoNan"div_no_nan_1(1333333�?9333333�?A333333�?I333333�?ao�G�$�6?i�� t���?�Unknown
v>HostAssignAddVariableOp"AssignAddVariableOp_4(1�������?9�������?A�������?I�������?a&���!�4?i026ؒ��?�Unknown
`?HostDivNoNan"
div_no_nan(1�������?9�������?A�������?I�������?a&���!�4?ik�k<(��?�Unknown
X@HostCast"Cast_2(1      �?9      �?A      �?I      �?a��;��2?i��>����?�Unknown
wAHostReadVariableOp"div_no_nan_1/ReadVariableOp(1�������?9�������?A�������?I�������?a�޵��0?i�h�����?�Unknown
�BHostReadVariableOp"+sequential_1/dense_5/BiasAdd/ReadVariableOp(1�������?9�������?A�������?I�������?a�޵��0?i^�!ǻ��?�Unknown
�CHostDivNoNan"3sparse_categorical_crossentropy/weighted_loss/value(1�������?9�������?A�������?I�������?a�޵��0?iV�����?�Unknown
vDHostAssignAddVariableOp"AssignAddVariableOp_3(1�������?9�������?A�������?I�������?a��_�0.?iܢ����?�Unknown
XEHostCast"Cast_3(1�������?9�������?A�������?I�������?a��_�0.?ib����?�Unknown
aFHostIdentity"Identity(1�������?9�������?A�������?I�������?a��_�0.?i���{��?�Unknown�
wGHostReadVariableOp"div_no_nan/ReadVariableOp_1(1�������?9�������?A�������?I�������?a��_�0.?i
n��\��?�Unknown
�HHostReadVariableOp"*sequential_1/dense_3/MatMul/ReadVariableOp(1ffffff�?9ffffff�?Affffff�?Iffffff�?a�S�*N*?iG���?�Unknown
uIHostReadVariableOp"div_no_nan/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?ao�G�$�&?iħʛj��?�Unknown
�JHostReadVariableOp"+sequential_1/dense_4/BiasAdd/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?ao�G�$�&?iAL^���?�Unknown
yKHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1      �?9      �?A      �?I      �?a��;��"?i�������?�Unknown