??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
?
	MLCConv2D

input"T
filter"T

unique_key"T*num_args
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)
"
	transposebool( "
num_argsint(
?
	MLCMatMul
a"T
b"T

unique_key"T*num_args
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2"
num_argsint ("

input_rankint(0
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*	2.4.0-rc02v1.12.1-44680-gc3fea33a218˪
?
block1_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameblock1_conv1/kernel
?
'block1_conv1/kernel/Read/ReadVariableOpReadVariableOpblock1_conv1/kernel*&
_output_shapes
:@*
dtype0
z
block1_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameblock1_conv1/bias
s
%block1_conv1/bias/Read/ReadVariableOpReadVariableOpblock1_conv1/bias*
_output_shapes
:@*
dtype0
?
block1_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*$
shared_nameblock1_conv2/kernel
?
'block1_conv2/kernel/Read/ReadVariableOpReadVariableOpblock1_conv2/kernel*&
_output_shapes
:@@*
dtype0
z
block1_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameblock1_conv2/bias
s
%block1_conv2/bias/Read/ReadVariableOpReadVariableOpblock1_conv2/bias*
_output_shapes
:@*
dtype0
?
block2_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*$
shared_nameblock2_conv1/kernel
?
'block2_conv1/kernel/Read/ReadVariableOpReadVariableOpblock2_conv1/kernel*'
_output_shapes
:@?*
dtype0
{
block2_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameblock2_conv1/bias
t
%block2_conv1/bias/Read/ReadVariableOpReadVariableOpblock2_conv1/bias*
_output_shapes	
:?*
dtype0
?
block2_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*$
shared_nameblock2_conv2/kernel
?
'block2_conv2/kernel/Read/ReadVariableOpReadVariableOpblock2_conv2/kernel*(
_output_shapes
:??*
dtype0
{
block2_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameblock2_conv2/bias
t
%block2_conv2/bias/Read/ReadVariableOpReadVariableOpblock2_conv2/bias*
_output_shapes	
:?*
dtype0
?
block3_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*$
shared_nameblock3_conv1/kernel
?
'block3_conv1/kernel/Read/ReadVariableOpReadVariableOpblock3_conv1/kernel*(
_output_shapes
:??*
dtype0
{
block3_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameblock3_conv1/bias
t
%block3_conv1/bias/Read/ReadVariableOpReadVariableOpblock3_conv1/bias*
_output_shapes	
:?*
dtype0
?
block3_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*$
shared_nameblock3_conv2/kernel
?
'block3_conv2/kernel/Read/ReadVariableOpReadVariableOpblock3_conv2/kernel*(
_output_shapes
:??*
dtype0
{
block3_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameblock3_conv2/bias
t
%block3_conv2/bias/Read/ReadVariableOpReadVariableOpblock3_conv2/bias*
_output_shapes	
:?*
dtype0
?
block3_conv3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*$
shared_nameblock3_conv3/kernel
?
'block3_conv3/kernel/Read/ReadVariableOpReadVariableOpblock3_conv3/kernel*(
_output_shapes
:??*
dtype0
{
block3_conv3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameblock3_conv3/bias
t
%block3_conv3/bias/Read/ReadVariableOpReadVariableOpblock3_conv3/bias*
_output_shapes	
:?*
dtype0
?
block4_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*$
shared_nameblock4_conv1/kernel
?
'block4_conv1/kernel/Read/ReadVariableOpReadVariableOpblock4_conv1/kernel*(
_output_shapes
:??*
dtype0
{
block4_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameblock4_conv1/bias
t
%block4_conv1/bias/Read/ReadVariableOpReadVariableOpblock4_conv1/bias*
_output_shapes	
:?*
dtype0
?
block4_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*$
shared_nameblock4_conv2/kernel
?
'block4_conv2/kernel/Read/ReadVariableOpReadVariableOpblock4_conv2/kernel*(
_output_shapes
:??*
dtype0
{
block4_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameblock4_conv2/bias
t
%block4_conv2/bias/Read/ReadVariableOpReadVariableOpblock4_conv2/bias*
_output_shapes	
:?*
dtype0
?
block4_conv3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*$
shared_nameblock4_conv3/kernel
?
'block4_conv3/kernel/Read/ReadVariableOpReadVariableOpblock4_conv3/kernel*(
_output_shapes
:??*
dtype0
{
block4_conv3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameblock4_conv3/bias
t
%block4_conv3/bias/Read/ReadVariableOpReadVariableOpblock4_conv3/bias*
_output_shapes	
:?*
dtype0
?
block5_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*$
shared_nameblock5_conv1/kernel
?
'block5_conv1/kernel/Read/ReadVariableOpReadVariableOpblock5_conv1/kernel*(
_output_shapes
:??*
dtype0
{
block5_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameblock5_conv1/bias
t
%block5_conv1/bias/Read/ReadVariableOpReadVariableOpblock5_conv1/bias*
_output_shapes	
:?*
dtype0

conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2d/kernel
x
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*'
_output_shapes
:?*
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
:*
dtype0
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?2*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	?2*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:2*
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:2*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:2*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:*
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0

NoOpNoOp
?]
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?\
value?\B?\ B?\
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer_with_weights-6

layer-9
layer-10
layer_with_weights-7
layer-11
layer_with_weights-8
layer-12
layer_with_weights-9
layer-13
layer-14
layer_with_weights-10
layer-15
layer_with_weights-11
layer-16
trainable_variables
	variables
regularization_losses
	keras_api

signatures
 
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
 	variables
!regularization_losses
"	keras_api
R
#trainable_variables
$	variables
%regularization_losses
&	keras_api
h

'kernel
(bias
)trainable_variables
*	variables
+regularization_losses
,	keras_api
h

-kernel
.bias
/trainable_variables
0	variables
1regularization_losses
2	keras_api
R
3trainable_variables
4	variables
5regularization_losses
6	keras_api
h

7kernel
8bias
9trainable_variables
:	variables
;regularization_losses
<	keras_api
h

=kernel
>bias
?trainable_variables
@	variables
Aregularization_losses
B	keras_api
h

Ckernel
Dbias
Etrainable_variables
F	variables
Gregularization_losses
H	keras_api
R
Itrainable_variables
J	variables
Kregularization_losses
L	keras_api
h

Mkernel
Nbias
Otrainable_variables
P	variables
Qregularization_losses
R	keras_api
h

Skernel
Tbias
Utrainable_variables
V	variables
Wregularization_losses
X	keras_api
h

Ykernel
Zbias
[trainable_variables
\	variables
]regularization_losses
^	keras_api
R
_trainable_variables
`	variables
aregularization_losses
b	keras_api
h

ckernel
dbias
etrainable_variables
f	variables
gregularization_losses
h	keras_api
?
ilayer_with_weights-0
ilayer-0
jlayer-1
klayer-2
llayer-3
mlayer_with_weights-1
mlayer-4
nlayer_with_weights-2
nlayer-5
o	optimizer
p
signatures
#q_self_saveable_object_factories
rtrainable_variables
s	variables
tregularization_losses
u	keras_api
?
0
1
2
3
'4
(5
-6
.7
78
89
=10
>11
C12
D13
M14
N15
S16
T17
Y18
Z19
c20
d21
v22
w23
x24
y25
z26
{27
?
0
1
2
3
'4
(5
-6
.7
78
89
=10
>11
C12
D13
M14
N15
S16
T17
Y18
Z19
c20
d21
v22
w23
x24
y25
z26
{27
 
?

|layers
trainable_variables
}layer_regularization_losses
~metrics
layer_metrics
?non_trainable_variables
	variables
regularization_losses
 
_]
VARIABLE_VALUEblock1_conv1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEblock1_conv1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
?layers
trainable_variables
 ?layer_regularization_losses
?metrics
?layer_metrics
?non_trainable_variables
	variables
regularization_losses
_]
VARIABLE_VALUEblock1_conv2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEblock1_conv2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
?layers
trainable_variables
 ?layer_regularization_losses
?metrics
?layer_metrics
?non_trainable_variables
 	variables
!regularization_losses
 
 
 
?
?layers
#trainable_variables
 ?layer_regularization_losses
?metrics
?layer_metrics
?non_trainable_variables
$	variables
%regularization_losses
_]
VARIABLE_VALUEblock2_conv1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEblock2_conv1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

'0
(1

'0
(1
 
?
?layers
)trainable_variables
 ?layer_regularization_losses
?metrics
?layer_metrics
?non_trainable_variables
*	variables
+regularization_losses
_]
VARIABLE_VALUEblock2_conv2/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEblock2_conv2/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

-0
.1

-0
.1
 
?
?layers
/trainable_variables
 ?layer_regularization_losses
?metrics
?layer_metrics
?non_trainable_variables
0	variables
1regularization_losses
 
 
 
?
?layers
3trainable_variables
 ?layer_regularization_losses
?metrics
?layer_metrics
?non_trainable_variables
4	variables
5regularization_losses
_]
VARIABLE_VALUEblock3_conv1/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEblock3_conv1/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

70
81

70
81
 
?
?layers
9trainable_variables
 ?layer_regularization_losses
?metrics
?layer_metrics
?non_trainable_variables
:	variables
;regularization_losses
_]
VARIABLE_VALUEblock3_conv2/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEblock3_conv2/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

=0
>1

=0
>1
 
?
?layers
?trainable_variables
 ?layer_regularization_losses
?metrics
?layer_metrics
?non_trainable_variables
@	variables
Aregularization_losses
_]
VARIABLE_VALUEblock3_conv3/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEblock3_conv3/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

C0
D1

C0
D1
 
?
?layers
Etrainable_variables
 ?layer_regularization_losses
?metrics
?layer_metrics
?non_trainable_variables
F	variables
Gregularization_losses
 
 
 
?
?layers
Itrainable_variables
 ?layer_regularization_losses
?metrics
?layer_metrics
?non_trainable_variables
J	variables
Kregularization_losses
_]
VARIABLE_VALUEblock4_conv1/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEblock4_conv1/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

M0
N1

M0
N1
 
?
?layers
Otrainable_variables
 ?layer_regularization_losses
?metrics
?layer_metrics
?non_trainable_variables
P	variables
Qregularization_losses
_]
VARIABLE_VALUEblock4_conv2/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEblock4_conv2/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

S0
T1

S0
T1
 
?
?layers
Utrainable_variables
 ?layer_regularization_losses
?metrics
?layer_metrics
?non_trainable_variables
V	variables
Wregularization_losses
_]
VARIABLE_VALUEblock4_conv3/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEblock4_conv3/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE

Y0
Z1

Y0
Z1
 
?
?layers
[trainable_variables
 ?layer_regularization_losses
?metrics
?layer_metrics
?non_trainable_variables
\	variables
]regularization_losses
 
 
 
?
?layers
_trainable_variables
 ?layer_regularization_losses
?metrics
?layer_metrics
?non_trainable_variables
`	variables
aregularization_losses
`^
VARIABLE_VALUEblock5_conv1/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEblock5_conv1/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE

c0
d1

c0
d1
 
?
?layers
etrainable_variables
 ?layer_regularization_losses
?metrics
?layer_metrics
?non_trainable_variables
f	variables
gregularization_losses
?

vkernel
wbias
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
|
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
|
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
|
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?

xkernel
ybias
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?

zkernel
{bias
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
 
 
 
*
v0
w1
x2
y3
z4
{5
*
v0
w1
x2
y3
z4
{5
 
?
?layers
rtrainable_variables
 ?layer_regularization_losses
?metrics
?layer_metrics
?non_trainable_variables
s	variables
tregularization_losses
TR
VARIABLE_VALUEconv2d/kernel1trainable_variables/22/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEconv2d/bias1trainable_variables/23/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEdense/kernel1trainable_variables/24/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUE
dense/bias1trainable_variables/25/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdense_1/kernel1trainable_variables/26/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEdense_1/bias1trainable_variables/27/.ATTRIBUTES/VARIABLE_VALUE
~
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

v0
w1

v0
w1
 
?
?layers
?trainable_variables
 ?layer_regularization_losses
?metrics
?layer_metrics
?non_trainable_variables
?	variables
?regularization_losses
 
 
 
 
?
?layers
?trainable_variables
 ?layer_regularization_losses
?metrics
?layer_metrics
?non_trainable_variables
?	variables
?regularization_losses
 
 
 
 
?
?layers
?trainable_variables
 ?layer_regularization_losses
?metrics
?layer_metrics
?non_trainable_variables
?	variables
?regularization_losses
 
 
 
 
?
?layers
?trainable_variables
 ?layer_regularization_losses
?metrics
?layer_metrics
?non_trainable_variables
?	variables
?regularization_losses
 

x0
y1

x0
y1
 
?
?layers
?trainable_variables
 ?layer_regularization_losses
?metrics
?layer_metrics
?non_trainable_variables
?	variables
?regularization_losses
 

z0
{1

z0
{1
 
?
?layers
?trainable_variables
 ?layer_regularization_losses
?metrics
?layer_metrics
?non_trainable_variables
?	variables
?regularization_losses
*
i0
j1
k2
l3
m4
n5
 

?0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
ge
VARIABLE_VALUEtotal_1Jlayer_with_weights-11/keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUEcount_1Jlayer_with_weights-11/keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
?
serving_default_input_1Placeholder*/
_output_shapes
:?????????xx*
dtype0*$
shape:?????????xx
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1block1_conv1/kernelblock1_conv1/biasblock1_conv2/kernelblock1_conv2/biasblock2_conv1/kernelblock2_conv1/biasblock2_conv2/kernelblock2_conv2/biasblock3_conv1/kernelblock3_conv1/biasblock3_conv2/kernelblock3_conv2/biasblock3_conv3/kernelblock3_conv3/biasblock4_conv1/kernelblock4_conv1/biasblock4_conv2/kernelblock4_conv2/biasblock4_conv3/kernelblock4_conv3/biasblock5_conv1/kernelblock5_conv1/biasconv2d/kernelconv2d/biasdense/kernel
dense/biasdense_1/kerneldense_1/bias*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*>
_read_only_resource_inputs 
	
*-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference_signature_wrapper_2213
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename'block1_conv1/kernel/Read/ReadVariableOp%block1_conv1/bias/Read/ReadVariableOp'block1_conv2/kernel/Read/ReadVariableOp%block1_conv2/bias/Read/ReadVariableOp'block2_conv1/kernel/Read/ReadVariableOp%block2_conv1/bias/Read/ReadVariableOp'block2_conv2/kernel/Read/ReadVariableOp%block2_conv2/bias/Read/ReadVariableOp'block3_conv1/kernel/Read/ReadVariableOp%block3_conv1/bias/Read/ReadVariableOp'block3_conv2/kernel/Read/ReadVariableOp%block3_conv2/bias/Read/ReadVariableOp'block3_conv3/kernel/Read/ReadVariableOp%block3_conv3/bias/Read/ReadVariableOp'block4_conv1/kernel/Read/ReadVariableOp%block4_conv1/bias/Read/ReadVariableOp'block4_conv2/kernel/Read/ReadVariableOp%block4_conv2/bias/Read/ReadVariableOp'block4_conv3/kernel/Read/ReadVariableOp%block4_conv3/bias/Read/ReadVariableOp'block5_conv1/kernel/Read/ReadVariableOp%block5_conv1/bias/Read/ReadVariableOp!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOpConst*+
Tin$
"2 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *&
f!R
__inference__traced_save_3085
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameblock1_conv1/kernelblock1_conv1/biasblock1_conv2/kernelblock1_conv2/biasblock2_conv1/kernelblock2_conv1/biasblock2_conv2/kernelblock2_conv2/biasblock3_conv1/kernelblock3_conv1/biasblock3_conv2/kernelblock3_conv2/biasblock3_conv3/kernelblock3_conv3/biasblock4_conv1/kernelblock4_conv1/biasblock4_conv2/kernelblock4_conv2/biasblock4_conv3/kernelblock4_conv3/biasblock5_conv1/kernelblock5_conv1/biasconv2d/kernelconv2d/biasdense/kernel
dense/biasdense_1/kerneldense_1/biastotal_1count_1**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__traced_restore_3185??
?
?
F__inference_block5_conv1_layer_call_and_return_conditional_losses_2698

inputs-
)conv2d_readvariableop_block5_conv1_kernel,
(biasadd_readvariableop_block5_conv1_bias
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOp)conv2d_readvariableop_block5_conv1_kernel*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2D	MLCConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
data_formatNCHW*
num_args *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOp(biasadd_readvariableop_block5_conv1_bias*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
data_formatNCHW2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?.
?
__inference__wrapped_model_423
input_29
5sequential_conv2d_conv2d_readvariableop_conv2d_kernel8
4sequential_conv2d_biasadd_readvariableop_conv2d_bias:
6sequential_dense_mlcmatmul_readvariableop_dense_kernel6
2sequential_dense_biasadd_readvariableop_dense_bias>
:sequential_dense_1_mlcmatmul_readvariableop_dense_1_kernel:
6sequential_dense_1_biasadd_readvariableop_dense_1_bias
identity??(sequential/conv2d/BiasAdd/ReadVariableOp?'sequential/conv2d/Conv2D/ReadVariableOp?'sequential/dense/BiasAdd/ReadVariableOp?)sequential/dense/MLCMatMul/ReadVariableOp?)sequential/dense_1/BiasAdd/ReadVariableOp?+sequential/dense_1/MLCMatMul/ReadVariableOp?
'sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp5sequential_conv2d_conv2d_readvariableop_conv2d_kernel*'
_output_shapes
:?*
dtype02)
'sequential/conv2d/Conv2D/ReadVariableOp?
sequential/conv2d/Conv2D	MLCConv2Dinput_2/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
data_formatNCHW*
num_args *
paddingVALID*
strides
2
sequential/conv2d/Conv2D?
(sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp4sequential_conv2d_biasadd_readvariableop_conv2d_bias*
_output_shapes
:*
dtype02*
(sequential/conv2d/BiasAdd/ReadVariableOp?
sequential/conv2d/BiasAddBiasAdd!sequential/conv2d/Conv2D:output:00sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
data_formatNCHW2
sequential/conv2d/BiasAdd?
sequential/conv2d/ReluRelu"sequential/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
sequential/conv2d/Relu?
 sequential/max_pooling2d/MaxPoolMaxPool$sequential/conv2d/Relu:activations:0*/
_output_shapes
:?????????*
data_formatNCHW*
ksize
*
paddingVALID*
strides
2"
 sequential/max_pooling2d/MaxPool?
!sequential/flatten/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!sequential/flatten/transpose/perm?
sequential/flatten/transpose	Transpose)sequential/max_pooling2d/MaxPool:output:0*sequential/flatten/transpose/perm:output:0*
T0*/
_output_shapes
:?????????2
sequential/flatten/transpose?
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????   2
sequential/flatten/Const?
sequential/flatten/ReshapeReshape sequential/flatten/transpose:y:0!sequential/flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
sequential/flatten/Reshape?
sequential/dropout/IdentityIdentity#sequential/flatten/Reshape:output:0*
T0*(
_output_shapes
:??????????2
sequential/dropout/Identity?
)sequential/dense/MLCMatMul/ReadVariableOpReadVariableOp6sequential_dense_mlcmatmul_readvariableop_dense_kernel*
_output_shapes
:	?2*
dtype02+
)sequential/dense/MLCMatMul/ReadVariableOp?
sequential/dense/MLCMatMul	MLCMatMul$sequential/dropout/Identity:output:01sequential/dense/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
sequential/dense/MLCMatMul?
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_biasadd_readvariableop_dense_bias*
_output_shapes
:2*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOp?
sequential/dense/BiasAddBiasAdd$sequential/dense/MLCMatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
sequential/dense/BiasAdd?
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22
sequential/dense/Relu?
+sequential/dense_1/MLCMatMul/ReadVariableOpReadVariableOp:sequential_dense_1_mlcmatmul_readvariableop_dense_1_kernel*
_output_shapes

:2*
dtype02-
+sequential/dense_1/MLCMatMul/ReadVariableOp?
sequential/dense_1/MLCMatMul	MLCMatMul#sequential/dense/Relu:activations:03sequential/dense_1/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential/dense_1/MLCMatMul?
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp6sequential_dense_1_biasadd_readvariableop_dense_1_bias*
_output_shapes
:*
dtype02+
)sequential/dense_1/BiasAdd/ReadVariableOp?
sequential/dense_1/BiasAddBiasAdd&sequential/dense_1/MLCMatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential/dense_1/BiasAdd?
sequential/dense_1/SigmoidSigmoid#sequential/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
sequential/dense_1/Sigmoid?
IdentityIdentitysequential/dense_1/Sigmoid:y:0)^sequential/conv2d/BiasAdd/ReadVariableOp(^sequential/conv2d/Conv2D/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp*^sequential/dense/MLCMatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp,^sequential/dense_1/MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:??????????::::::2T
(sequential/conv2d/BiasAdd/ReadVariableOp(sequential/conv2d/BiasAdd/ReadVariableOp2R
'sequential/conv2d/Conv2D/ReadVariableOp'sequential/conv2d/Conv2D/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2V
)sequential/dense/MLCMatMul/ReadVariableOp)sequential/dense/MLCMatMul/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2Z
+sequential/dense_1/MLCMatMul/ReadVariableOp+sequential/dense_1/MLCMatMul/ReadVariableOp:Y U
0
_output_shapes
:??????????
!
_user_specified_name	input_2
?
]
A__inference_flatten_layer_call_and_return_conditional_losses_2907

inputs
identityy
transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
transpose/perm~
	transpose	Transposeinputstranspose/perm:output:0*
T0*/
_output_shapes
:?????????2
	transpose_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????   2
Consto
ReshapeReshapetranspose:y:0Const:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
a
E__inference_block4_pool_layer_call_and_return_conditional_losses_1375

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
data_formatNCHW*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
)__inference_sequential_layer_call_fn_2881

inputs

kernel
bias
kernel_1

bias_1
kernel_2

bias_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputskernelbiaskernel_1bias_1kernel_2bias_2*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_15812
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:??????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
F
*__inference_block1_pool_layer_call_fn_1327

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_block1_pool_layer_call_and_return_conditional_losses_13242
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
+__inference_block1_conv1_layer_call_fn_2525

inputs
block1_conv1_kernel
block1_conv1_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsblock1_conv1_kernelblock1_conv1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@xx*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block1_conv1_layer_call_and_return_conditional_losses_16052
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????@xx2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????xx::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????xx
 
_user_specified_nameinputs
?
c
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_1392

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
data_formatNCHW*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
]
A__inference_dropout_layer_call_and_return_conditional_losses_1454

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/Mula
dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
dropout/rate^
dropout/seedConst*
_output_shapes
: *
dtype0*
value	B : 2
dropout/seed?
dropout
MLCDropoutinputsdropout/rate:output:0dropout/seed:output:0*
T0*(
_output_shapes
:??????????2	
dropoute
IdentityIdentitydropout:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?%
?
D__inference_sequential_layer_call_and_return_conditional_losses_2859

inputs'
#conv2d_conv2d_readvariableop_kernel&
"conv2d_biasadd_readvariableop_bias+
'dense_mlcmatmul_readvariableop_kernel_1'
#dense_biasadd_readvariableop_bias_1-
)dense_1_mlcmatmul_readvariableop_kernel_2)
%dense_1_biasadd_readvariableop_bias_2
identity??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MLCMatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp? dense_1/MLCMatMul/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp#conv2d_conv2d_readvariableop_kernel*'
_output_shapes
:?*
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2D	MLCConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
data_formatNCHW*
num_args *
paddingVALID*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp"conv2d_biasadd_readvariableop_bias*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
data_formatNCHW2
conv2d/BiasAddu
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d/Relu?
max_pooling2d/MaxPoolMaxPoolconv2d/Relu:activations:0*/
_output_shapes
:?????????*
data_formatNCHW*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool?
flatten/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
flatten/transpose/perm?
flatten/transpose	Transposemax_pooling2d/MaxPool:output:0flatten/transpose/perm:output:0*
T0*/
_output_shapes
:?????????2
flatten/transposeo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????   2
flatten/Const?
flatten/ReshapeReshapeflatten/transpose:y:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten/Reshape}
dropout/IdentityIdentityflatten/Reshape:output:0*
T0*(
_output_shapes
:??????????2
dropout/Identity?
dense/MLCMatMul/ReadVariableOpReadVariableOp'dense_mlcmatmul_readvariableop_kernel_1*
_output_shapes
:	?2*
dtype02 
dense/MLCMatMul/ReadVariableOp?
dense/MLCMatMul	MLCMatMuldropout/Identity:output:0&dense/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
dense/MLCMatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp#dense_biasadd_readvariableop_bias_1*
_output_shapes
:2*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MLCMatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
dense/BiasAddj

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22

dense/Relu?
 dense_1/MLCMatMul/ReadVariableOpReadVariableOp)dense_1_mlcmatmul_readvariableop_kernel_2*
_output_shapes

:2*
dtype02"
 dense_1/MLCMatMul/ReadVariableOp?
dense_1/MLCMatMul	MLCMatMuldense/Relu:activations:0(dense_1/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/MLCMatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp%dense_1_biasadd_readvariableop_bias_2*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MLCMatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/BiasAddy
dense_1/SigmoidSigmoiddense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_1/Sigmoid?
IdentityIdentitydense_1/Sigmoid:y:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MLCMatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp!^dense_1/MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:??????????::::::2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/MLCMatMul/ReadVariableOpdense/MLCMatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2D
 dense_1/MLCMatMul/ReadVariableOp dense_1/MLCMatMul/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?%
?
D__inference_sequential_layer_call_and_return_conditional_losses_2771
inputs_0'
#conv2d_conv2d_readvariableop_kernel&
"conv2d_biasadd_readvariableop_bias+
'dense_mlcmatmul_readvariableop_kernel_1'
#dense_biasadd_readvariableop_bias_1-
)dense_1_mlcmatmul_readvariableop_kernel_2)
%dense_1_biasadd_readvariableop_bias_2
identity??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MLCMatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp? dense_1/MLCMatMul/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp#conv2d_conv2d_readvariableop_kernel*'
_output_shapes
:?*
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2D	MLCConv2Dinputs_0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
data_formatNCHW*
num_args *
paddingVALID*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp"conv2d_biasadd_readvariableop_bias*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
data_formatNCHW2
conv2d/BiasAddu
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d/Relu?
max_pooling2d/MaxPoolMaxPoolconv2d/Relu:activations:0*/
_output_shapes
:?????????*
data_formatNCHW*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool?
flatten/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
flatten/transpose/perm?
flatten/transpose	Transposemax_pooling2d/MaxPool:output:0flatten/transpose/perm:output:0*
T0*/
_output_shapes
:?????????2
flatten/transposeo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????   2
flatten/Const?
flatten/ReshapeReshapeflatten/transpose:y:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten/Reshape}
dropout/IdentityIdentityflatten/Reshape:output:0*
T0*(
_output_shapes
:??????????2
dropout/Identity?
dense/MLCMatMul/ReadVariableOpReadVariableOp'dense_mlcmatmul_readvariableop_kernel_1*
_output_shapes
:	?2*
dtype02 
dense/MLCMatMul/ReadVariableOp?
dense/MLCMatMul	MLCMatMuldropout/Identity:output:0&dense/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
dense/MLCMatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp#dense_biasadd_readvariableop_bias_1*
_output_shapes
:2*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MLCMatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
dense/BiasAddj

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22

dense/Relu?
 dense_1/MLCMatMul/ReadVariableOpReadVariableOp)dense_1_mlcmatmul_readvariableop_kernel_2*
_output_shapes

:2*
dtype02"
 dense_1/MLCMatMul/ReadVariableOp?
dense_1/MLCMatMul	MLCMatMuldense/Relu:activations:0(dense_1/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/MLCMatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp%dense_1_biasadd_readvariableop_bias_2*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MLCMatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/BiasAddy
dense_1/SigmoidSigmoiddense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_1/Sigmoid?
IdentityIdentitydense_1/Sigmoid:y:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MLCMatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp!^dense_1/MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:??????????::::::2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/MLCMatMul/ReadVariableOpdense/MLCMatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2D
 dense_1/MLCMatMul/ReadVariableOp dense_1/MLCMatMul/ReadVariableOp:Z V
0
_output_shapes
:??????????
"
_user_specified_name
inputs/0
?
]
A__inference_flatten_layer_call_and_return_conditional_losses_1436

inputs
identityy
transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
transpose/perm~
	transpose	Transposeinputstranspose/perm:output:0*
T0*/
_output_shapes
:?????????2
	transpose_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????   2
Consto
ReshapeReshapetranspose:y:0Const:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
B
&__inference_dropout_layer_call_fn_2931

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_14542
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
?__inference_dense_layer_call_and_return_conditional_losses_1483

inputs%
!mlcmatmul_readvariableop_kernel_1!
biasadd_readvariableop_bias_1
identity??BiasAdd/ReadVariableOp?MLCMatMul/ReadVariableOp?
MLCMatMul/ReadVariableOpReadVariableOp!mlcmatmul_readvariableop_kernel_1*
_output_shapes
:	?2*
dtype02
MLCMatMul/ReadVariableOp
	MLCMatMul	MLCMatMulinputs MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
	MLCMatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_bias_1*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMLCMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????22
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
MLCMatMul/ReadVariableOpMLCMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
a
E__inference_block3_pool_layer_call_and_return_conditional_losses_1358

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
data_formatNCHW*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
t
%__inference_conv2d_layer_call_fn_2899

inputs

kernel
bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputskernelbias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_14102
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
D__inference_sequential_layer_call_and_return_conditional_losses_1519
input_2
conv2d_kernel
conv2d_bias
dense_kernel_1
dense_bias_1
dense_1_kernel_2
dense_1_bias_2
identity??conv2d/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_2conv2d_kernelconv2d_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_14102 
conv2d/StatefulPartitionedCall?
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_13922
max_pooling2d/PartitionedCall?
flatten/PartitionedCallPartitionedCall&max_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_14362
flatten/PartitionedCall?
dropout/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_14542
dropout/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_kernel_1dense_bias_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_14832
dense/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_kernel_2dense_1_bias_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_15062!
dense_1/StatefulPartitionedCall?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:??????????::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:Y U
0
_output_shapes
:??????????
!
_user_specified_name	input_2
?
?
)__inference_sequential_layer_call_fn_1563
input_2

kernel
bias
kernel_1

bias_1
kernel_2

bias_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_2kernelbiaskernel_1bias_1kernel_2bias_2*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_15542
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:??????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:??????????
!
_user_specified_name	input_2
?
?
)__inference_sequential_layer_call_fn_2870

inputs

kernel
bias
kernel_1

bias_1
kernel_2

bias_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputskernelbiaskernel_1bias_1kernel_2bias_2*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_15542
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:??????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
F__inference_block1_conv2_layer_call_and_return_conditional_losses_2536

inputs-
)conv2d_readvariableop_block1_conv2_kernel,
(biasadd_readvariableop_block1_conv2_bias
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOp)conv2d_readvariableop_block1_conv2_kernel*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2D	MLCConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@xx*
data_formatNCHW*
num_args *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOp(biasadd_readvariableop_block1_conv2_bias*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@xx*
data_formatNCHW2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@xx2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@xx2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@xx::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@xx
 
_user_specified_nameinputs
?
?
F__inference_block3_conv3_layer_call_and_return_conditional_losses_1755

inputs-
)conv2d_readvariableop_block3_conv3_kernel,
(biasadd_readvariableop_block3_conv3_bias
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOp)conv2d_readvariableop_block3_conv3_kernel*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2D	MLCConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
data_formatNCHW*
num_args *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOp(biasadd_readvariableop_block3_conv3_bias*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
data_formatNCHW2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
F__inference_block3_conv2_layer_call_and_return_conditional_losses_1732

inputs-
)conv2d_readvariableop_block3_conv2_kernel,
(biasadd_readvariableop_block3_conv2_bias
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOp)conv2d_readvariableop_block3_conv2_kernel*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2D	MLCConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
data_formatNCHW*
num_args *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOp(biasadd_readvariableop_block3_conv2_bias*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
data_formatNCHW2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
@__inference_conv2d_layer_call_and_return_conditional_losses_2892

inputs 
conv2d_readvariableop_kernel
biasadd_readvariableop_bias
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_kernel*'
_output_shapes
:?*
dtype02
Conv2D/ReadVariableOp?
Conv2D	MLCConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
data_formatNCHW*
num_args *
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_bias*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
data_formatNCHW2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
F__inference_block1_conv2_layer_call_and_return_conditional_losses_1628

inputs-
)conv2d_readvariableop_block1_conv2_kernel,
(biasadd_readvariableop_block1_conv2_bias
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOp)conv2d_readvariableop_block1_conv2_kernel*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2D	MLCConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@xx*
data_formatNCHW*
num_args *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOp(biasadd_readvariableop_block1_conv2_bias*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@xx*
data_formatNCHW2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@xx2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@xx2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@xx::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@xx
 
_user_specified_nameinputs
?
?
+__inference_block3_conv2_layer_call_fn_2615

inputs
block3_conv2_kernel
block3_conv2_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsblock3_conv2_kernelblock3_conv2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block3_conv2_layer_call_and_return_conditional_losses_17322
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?B
?
__inference__traced_save_3085
file_prefix2
.savev2_block1_conv1_kernel_read_readvariableop0
,savev2_block1_conv1_bias_read_readvariableop2
.savev2_block1_conv2_kernel_read_readvariableop0
,savev2_block1_conv2_bias_read_readvariableop2
.savev2_block2_conv1_kernel_read_readvariableop0
,savev2_block2_conv1_bias_read_readvariableop2
.savev2_block2_conv2_kernel_read_readvariableop0
,savev2_block2_conv2_bias_read_readvariableop2
.savev2_block3_conv1_kernel_read_readvariableop0
,savev2_block3_conv1_bias_read_readvariableop2
.savev2_block3_conv2_kernel_read_readvariableop0
,savev2_block3_conv2_bias_read_readvariableop2
.savev2_block3_conv3_kernel_read_readvariableop0
,savev2_block3_conv3_bias_read_readvariableop2
.savev2_block4_conv1_kernel_read_readvariableop0
,savev2_block4_conv1_bias_read_readvariableop2
.savev2_block4_conv2_kernel_read_readvariableop0
,savev2_block4_conv2_bias_read_readvariableop2
.savev2_block4_conv3_kernel_read_readvariableop0
,savev2_block4_conv3_bias_read_readvariableop2
.savev2_block5_conv1_kernel_read_readvariableop0
,savev2_block5_conv1_bias_read_readvariableop,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/22/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/23/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/24/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/25/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/26/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/27/.ATTRIBUTES/VARIABLE_VALUEBJlayer_with_weights-11/keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEBJlayer_with_weights-11/keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Q
valueHBFB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0.savev2_block1_conv1_kernel_read_readvariableop,savev2_block1_conv1_bias_read_readvariableop.savev2_block1_conv2_kernel_read_readvariableop,savev2_block1_conv2_bias_read_readvariableop.savev2_block2_conv1_kernel_read_readvariableop,savev2_block2_conv1_bias_read_readvariableop.savev2_block2_conv2_kernel_read_readvariableop,savev2_block2_conv2_bias_read_readvariableop.savev2_block3_conv1_kernel_read_readvariableop,savev2_block3_conv1_bias_read_readvariableop.savev2_block3_conv2_kernel_read_readvariableop,savev2_block3_conv2_bias_read_readvariableop.savev2_block3_conv3_kernel_read_readvariableop,savev2_block3_conv3_bias_read_readvariableop.savev2_block4_conv1_kernel_read_readvariableop,savev2_block4_conv1_bias_read_readvariableop.savev2_block4_conv2_kernel_read_readvariableop,savev2_block4_conv2_bias_read_readvariableop.savev2_block4_conv3_kernel_read_readvariableop,savev2_block4_conv3_bias_read_readvariableop.savev2_block5_conv1_kernel_read_readvariableop,savev2_block5_conv1_bias_read_readvariableop(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *-
dtypes#
!22
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :@:@:@@:@:@?:?:??:?:??:?:??:?:??:?:??:?:??:?:??:?:??:?:?::	?2:2:2:: : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:-)
'
_output_shapes
:@?:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:.	*
(
_output_shapes
:??:!


_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:-)
'
_output_shapes
:?: 

_output_shapes
::%!

_output_shapes
:	?2: 

_output_shapes
:2:$ 

_output_shapes

:2: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
)__inference_sequential_layer_call_fn_2782
inputs_0

kernel
bias
kernel_1

bias_1
kernel_2

bias_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0kernelbiaskernel_1bias_1kernel_2bias_2*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_19052
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:??????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
0
_output_shapes
:??????????
"
_user_specified_name
inputs/0
?

?
A__inference_dense_1_layer_call_and_return_conditional_losses_1506

inputs%
!mlcmatmul_readvariableop_kernel_2!
biasadd_readvariableop_bias_2
identity??BiasAdd/ReadVariableOp?MLCMatMul/ReadVariableOp?
MLCMatMul/ReadVariableOpReadVariableOp!mlcmatmul_readvariableop_kernel_2*
_output_shapes

:2*
dtype02
MLCMatMul/ReadVariableOp
	MLCMatMul	MLCMatMulinputs MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
	MLCMatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_bias_2*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMLCMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????2::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
MLCMatMul/ReadVariableOpMLCMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
w
$__inference_dense_layer_call_fn_2954

inputs
kernel_1

bias_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputskernel_1bias_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_14832
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
@__inference_conv2d_layer_call_and_return_conditional_losses_1410

inputs 
conv2d_readvariableop_kernel
biasadd_readvariableop_bias
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_kernel*'
_output_shapes
:?*
dtype02
Conv2D/ReadVariableOp?
Conv2D	MLCConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
data_formatNCHW*
num_args *
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_bias*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
data_formatNCHW2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
F__inference_block2_conv2_layer_call_and_return_conditional_losses_2572

inputs-
)conv2d_readvariableop_block2_conv2_kernel,
(biasadd_readvariableop_block2_conv2_bias
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOp)conv2d_readvariableop_block2_conv2_kernel*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2D	MLCConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????<<*
data_formatNCHW*
num_args *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOp(biasadd_readvariableop_block2_conv2_bias*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????<<*
data_formatNCHW2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????<<2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????<<2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????<<::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????<<
 
_user_specified_nameinputs
?
?
F__inference_block4_conv2_layer_call_and_return_conditional_losses_2662

inputs-
)conv2d_readvariableop_block4_conv2_kernel,
(biasadd_readvariableop_block4_conv2_bias
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOp)conv2d_readvariableop_block4_conv2_kernel*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2D	MLCConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
data_formatNCHW*
num_args *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOp(biasadd_readvariableop_block4_conv2_bias*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
data_formatNCHW2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
&__inference_model_1_layer_call_fn_2178
input_1
block1_conv1_kernel
block1_conv1_bias
block1_conv2_kernel
block1_conv2_bias
block2_conv1_kernel
block2_conv1_bias
block2_conv2_kernel
block2_conv2_bias
block3_conv1_kernel
block3_conv1_bias
block3_conv2_kernel
block3_conv2_bias
block3_conv3_kernel
block3_conv3_bias
block4_conv1_kernel
block4_conv1_bias
block4_conv2_kernel
block4_conv2_bias
block4_conv3_kernel
block4_conv3_bias
block5_conv1_kernel
block5_conv1_bias

kernel
bias
kernel_1

bias_1
kernel_2

bias_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1block1_conv1_kernelblock1_conv1_biasblock1_conv2_kernelblock1_conv2_biasblock2_conv1_kernelblock2_conv1_biasblock2_conv2_kernelblock2_conv2_biasblock3_conv1_kernelblock3_conv1_biasblock3_conv2_kernelblock3_conv2_biasblock3_conv3_kernelblock3_conv3_biasblock4_conv1_kernelblock4_conv1_biasblock4_conv2_kernelblock4_conv2_biasblock4_conv3_kernelblock4_conv3_biasblock5_conv1_kernelblock5_conv1_biaskernelbiaskernel_1bias_1kernel_2bias_2*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*>
_read_only_resource_inputs 
	
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_model_1_layer_call_and_return_conditional_losses_21472
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????xx::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????xx
!
_user_specified_name	input_1
?
?
F__inference_block4_conv2_layer_call_and_return_conditional_losses_1807

inputs-
)conv2d_readvariableop_block4_conv2_kernel,
(biasadd_readvariableop_block4_conv2_bias
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOp)conv2d_readvariableop_block4_conv2_kernel*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2D	MLCConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
data_formatNCHW*
num_args *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOp(biasadd_readvariableop_block4_conv2_bias*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
data_formatNCHW2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
F__inference_block3_conv3_layer_call_and_return_conditional_losses_2626

inputs-
)conv2d_readvariableop_block3_conv3_kernel,
(biasadd_readvariableop_block3_conv3_bias
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOp)conv2d_readvariableop_block3_conv3_kernel*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2D	MLCConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
data_formatNCHW*
num_args *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOp(biasadd_readvariableop_block3_conv3_bias*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
data_formatNCHW2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
&__inference_model_1_layer_call_fn_2474

inputs
block1_conv1_kernel
block1_conv1_bias
block1_conv2_kernel
block1_conv2_bias
block2_conv1_kernel
block2_conv1_bias
block2_conv2_kernel
block2_conv2_bias
block3_conv1_kernel
block3_conv1_bias
block3_conv2_kernel
block3_conv2_bias
block3_conv3_kernel
block3_conv3_bias
block4_conv1_kernel
block4_conv1_bias
block4_conv2_kernel
block4_conv2_bias
block4_conv3_kernel
block4_conv3_bias
block5_conv1_kernel
block5_conv1_bias

kernel
bias
kernel_1

bias_1
kernel_2

bias_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsblock1_conv1_kernelblock1_conv1_biasblock1_conv2_kernelblock1_conv2_biasblock2_conv1_kernelblock2_conv1_biasblock2_conv2_kernelblock2_conv2_biasblock3_conv1_kernelblock3_conv1_biasblock3_conv2_kernelblock3_conv2_biasblock3_conv3_kernelblock3_conv3_biasblock4_conv1_kernelblock4_conv1_biasblock4_conv2_kernelblock4_conv2_biasblock4_conv3_kernelblock4_conv3_biasblock5_conv1_kernelblock5_conv1_biaskernelbiaskernel_1bias_1kernel_2bias_2*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*>
_read_only_resource_inputs 
	
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_model_1_layer_call_and_return_conditional_losses_20662
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????xx::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????xx
 
_user_specified_nameinputs
?
?
D__inference_sequential_layer_call_and_return_conditional_losses_1554

inputs
conv2d_kernel
conv2d_bias
dense_kernel_1
dense_bias_1
dense_1_kernel_2
dense_1_bias_2
identity??conv2d/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_kernelconv2d_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_14102 
conv2d/StatefulPartitionedCall?
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_13922
max_pooling2d/PartitionedCall?
flatten/PartitionedCallPartitionedCall&max_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_14362
flatten/PartitionedCall?
dropout/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_14542
dropout/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_kernel_1dense_bias_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_14832
dense/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_kernel_2dense_1_bias_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_15062!
dense_1/StatefulPartitionedCall?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:??????????::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
D__inference_sequential_layer_call_and_return_conditional_losses_1581

inputs
conv2d_kernel
conv2d_bias
dense_kernel_1
dense_bias_1
dense_1_kernel_2
dense_1_bias_2
identity??conv2d/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_kernelconv2d_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_14102 
conv2d/StatefulPartitionedCall?
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_13922
max_pooling2d/PartitionedCall?
flatten/PartitionedCallPartitionedCall&max_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_14362
flatten/PartitionedCall?
dropout/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_14592
dropout/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_kernel_1dense_bias_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_14832
dense/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_kernel_2dense_1_bias_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_15062!
dense_1/StatefulPartitionedCall?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:??????????::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
)__inference_sequential_layer_call_fn_1590
input_2

kernel
bias
kernel_1

bias_1
kernel_2

bias_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_2kernelbiaskernel_1bias_1kernel_2bias_2*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_15812
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:??????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:??????????
!
_user_specified_name	input_2
?
F
*__inference_block4_pool_layer_call_fn_1378

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_block4_pool_layer_call_and_return_conditional_losses_13752
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?%
?
D__inference_sequential_layer_call_and_return_conditional_losses_1936

inputs'
#conv2d_conv2d_readvariableop_kernel&
"conv2d_biasadd_readvariableop_bias+
'dense_mlcmatmul_readvariableop_kernel_1'
#dense_biasadd_readvariableop_bias_1-
)dense_1_mlcmatmul_readvariableop_kernel_2)
%dense_1_biasadd_readvariableop_bias_2
identity??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MLCMatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp? dense_1/MLCMatMul/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp#conv2d_conv2d_readvariableop_kernel*'
_output_shapes
:?*
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2D	MLCConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
data_formatNCHW*
num_args *
paddingVALID*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp"conv2d_biasadd_readvariableop_bias*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
data_formatNCHW2
conv2d/BiasAddu
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d/Relu?
max_pooling2d/MaxPoolMaxPoolconv2d/Relu:activations:0*/
_output_shapes
:?????????*
data_formatNCHW*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool?
flatten/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
flatten/transpose/perm?
flatten/transpose	Transposemax_pooling2d/MaxPool:output:0flatten/transpose/perm:output:0*
T0*/
_output_shapes
:?????????2
flatten/transposeo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????   2
flatten/Const?
flatten/ReshapeReshapeflatten/transpose:y:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten/Reshape}
dropout/IdentityIdentityflatten/Reshape:output:0*
T0*(
_output_shapes
:??????????2
dropout/Identity?
dense/MLCMatMul/ReadVariableOpReadVariableOp'dense_mlcmatmul_readvariableop_kernel_1*
_output_shapes
:	?2*
dtype02 
dense/MLCMatMul/ReadVariableOp?
dense/MLCMatMul	MLCMatMuldropout/Identity:output:0&dense/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
dense/MLCMatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp#dense_biasadd_readvariableop_bias_1*
_output_shapes
:2*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MLCMatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
dense/BiasAddj

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22

dense/Relu?
 dense_1/MLCMatMul/ReadVariableOpReadVariableOp)dense_1_mlcmatmul_readvariableop_kernel_2*
_output_shapes

:2*
dtype02"
 dense_1/MLCMatMul/ReadVariableOp?
dense_1/MLCMatMul	MLCMatMuldense/Relu:activations:0(dense_1/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/MLCMatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp%dense_1_biasadd_readvariableop_bias_2*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MLCMatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/BiasAddy
dense_1/SigmoidSigmoiddense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_1/Sigmoid?
IdentityIdentitydense_1/Sigmoid:y:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MLCMatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp!^dense_1/MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:??????????::::::2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/MLCMatMul/ReadVariableOpdense/MLCMatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2D
 dense_1/MLCMatMul/ReadVariableOp dense_1/MLCMatMul/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
F
*__inference_block2_pool_layer_call_fn_1344

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_block2_pool_layer_call_and_return_conditional_losses_13412
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
B
&__inference_dropout_layer_call_fn_2936

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_14592
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
+__inference_block3_conv1_layer_call_fn_2597

inputs
block3_conv1_kernel
block3_conv1_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsblock3_conv1_kernelblock3_conv1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block3_conv1_layer_call_and_return_conditional_losses_17092
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?T
?
A__inference_model_1_layer_call_and_return_conditional_losses_1967
input_1$
 block1_conv1_block1_conv1_kernel"
block1_conv1_block1_conv1_bias$
 block1_conv2_block1_conv2_kernel"
block1_conv2_block1_conv2_bias$
 block2_conv1_block2_conv1_kernel"
block2_conv1_block2_conv1_bias$
 block2_conv2_block2_conv2_kernel"
block2_conv2_block2_conv2_bias$
 block3_conv1_block3_conv1_kernel"
block3_conv1_block3_conv1_bias$
 block3_conv2_block3_conv2_kernel"
block3_conv2_block3_conv2_bias$
 block3_conv3_block3_conv3_kernel"
block3_conv3_block3_conv3_bias$
 block4_conv1_block4_conv1_kernel"
block4_conv1_block4_conv1_bias$
 block4_conv2_block4_conv2_kernel"
block4_conv2_block4_conv2_bias$
 block4_conv3_block4_conv3_kernel"
block4_conv3_block4_conv3_bias$
 block5_conv1_block5_conv1_kernel"
block5_conv1_block5_conv1_bias
sequential_kernel
sequential_bias
sequential_kernel_1
sequential_bias_1
sequential_kernel_2
sequential_bias_2
identity??$block1_conv1/StatefulPartitionedCall?$block1_conv2/StatefulPartitionedCall?$block2_conv1/StatefulPartitionedCall?$block2_conv2/StatefulPartitionedCall?$block3_conv1/StatefulPartitionedCall?$block3_conv2/StatefulPartitionedCall?$block3_conv3/StatefulPartitionedCall?$block4_conv1/StatefulPartitionedCall?$block4_conv2/StatefulPartitionedCall?$block4_conv3/StatefulPartitionedCall?$block5_conv1/StatefulPartitionedCall?"sequential/StatefulPartitionedCall?
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCallinput_1 block1_conv1_block1_conv1_kernelblock1_conv1_block1_conv1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@xx*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block1_conv1_layer_call_and_return_conditional_losses_16052&
$block1_conv1/StatefulPartitionedCall?
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0 block1_conv2_block1_conv2_kernelblock1_conv2_block1_conv2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@xx*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block1_conv2_layer_call_and_return_conditional_losses_16282&
$block1_conv2/StatefulPartitionedCall?
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@<<* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_block1_pool_layer_call_and_return_conditional_losses_13242
block1_pool/PartitionedCall?
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall$block1_pool/PartitionedCall:output:0 block2_conv1_block2_conv1_kernelblock2_conv1_block2_conv1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????<<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block2_conv1_layer_call_and_return_conditional_losses_16572&
$block2_conv1/StatefulPartitionedCall?
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0 block2_conv2_block2_conv2_kernelblock2_conv2_block2_conv2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????<<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block2_conv2_layer_call_and_return_conditional_losses_16802&
$block2_conv2/StatefulPartitionedCall?
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_block2_pool_layer_call_and_return_conditional_losses_13412
block2_pool/PartitionedCall?
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall$block2_pool/PartitionedCall:output:0 block3_conv1_block3_conv1_kernelblock3_conv1_block3_conv1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block3_conv1_layer_call_and_return_conditional_losses_17092&
$block3_conv1/StatefulPartitionedCall?
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0 block3_conv2_block3_conv2_kernelblock3_conv2_block3_conv2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block3_conv2_layer_call_and_return_conditional_losses_17322&
$block3_conv2/StatefulPartitionedCall?
$block3_conv3/StatefulPartitionedCallStatefulPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0 block3_conv3_block3_conv3_kernelblock3_conv3_block3_conv3_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block3_conv3_layer_call_and_return_conditional_losses_17552&
$block3_conv3/StatefulPartitionedCall?
block3_pool/PartitionedCallPartitionedCall-block3_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_block3_pool_layer_call_and_return_conditional_losses_13582
block3_pool/PartitionedCall?
$block4_conv1/StatefulPartitionedCallStatefulPartitionedCall$block3_pool/PartitionedCall:output:0 block4_conv1_block4_conv1_kernelblock4_conv1_block4_conv1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block4_conv1_layer_call_and_return_conditional_losses_17842&
$block4_conv1/StatefulPartitionedCall?
$block4_conv2/StatefulPartitionedCallStatefulPartitionedCall-block4_conv1/StatefulPartitionedCall:output:0 block4_conv2_block4_conv2_kernelblock4_conv2_block4_conv2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block4_conv2_layer_call_and_return_conditional_losses_18072&
$block4_conv2/StatefulPartitionedCall?
$block4_conv3/StatefulPartitionedCallStatefulPartitionedCall-block4_conv2/StatefulPartitionedCall:output:0 block4_conv3_block4_conv3_kernelblock4_conv3_block4_conv3_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block4_conv3_layer_call_and_return_conditional_losses_18302&
$block4_conv3/StatefulPartitionedCall?
block4_pool/PartitionedCallPartitionedCall-block4_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_block4_pool_layer_call_and_return_conditional_losses_13752
block4_pool/PartitionedCall?
$block5_conv1/StatefulPartitionedCallStatefulPartitionedCall$block4_pool/PartitionedCall:output:0 block5_conv1_block5_conv1_kernelblock5_conv1_block5_conv1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block5_conv1_layer_call_and_return_conditional_losses_18592&
$block5_conv1/StatefulPartitionedCall?
"sequential/StatefulPartitionedCallStatefulPartitionedCall-block5_conv1/StatefulPartitionedCall:output:0sequential_kernelsequential_biassequential_kernel_1sequential_bias_1sequential_kernel_2sequential_bias_2*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_19052$
"sequential/StatefulPartitionedCall?
IdentityIdentity+sequential/StatefulPartitionedCall:output:0%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall#^sequential/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????xx::::::::::::::::::::::::::::2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall2L
$block3_conv2/StatefulPartitionedCall$block3_conv2/StatefulPartitionedCall2L
$block3_conv3/StatefulPartitionedCall$block3_conv3/StatefulPartitionedCall2L
$block4_conv1/StatefulPartitionedCall$block4_conv1/StatefulPartitionedCall2L
$block4_conv2/StatefulPartitionedCall$block4_conv2/StatefulPartitionedCall2L
$block4_conv3/StatefulPartitionedCall$block4_conv3/StatefulPartitionedCall2L
$block5_conv1/StatefulPartitionedCall$block5_conv1/StatefulPartitionedCall2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall:X T
/
_output_shapes
:?????????xx
!
_user_specified_name	input_1
?)
?
D__inference_sequential_layer_call_and_return_conditional_losses_1905

inputs'
#conv2d_conv2d_readvariableop_kernel&
"conv2d_biasadd_readvariableop_bias+
'dense_mlcmatmul_readvariableop_kernel_1'
#dense_biasadd_readvariableop_bias_1-
)dense_1_mlcmatmul_readvariableop_kernel_2)
%dense_1_biasadd_readvariableop_bias_2
identity??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MLCMatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp? dense_1/MLCMatMul/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp#conv2d_conv2d_readvariableop_kernel*'
_output_shapes
:?*
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2D	MLCConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
data_formatNCHW*
num_args *
paddingVALID*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp"conv2d_biasadd_readvariableop_bias*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
data_formatNCHW2
conv2d/BiasAddu
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d/Relu?
max_pooling2d/MaxPoolMaxPoolconv2d/Relu:activations:0*/
_output_shapes
:?????????*
data_formatNCHW*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool?
flatten/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
flatten/transpose/perm?
flatten/transpose	Transposemax_pooling2d/MaxPool:output:0flatten/transpose/perm:output:0*
T0*/
_output_shapes
:?????????2
flatten/transposeo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????   2
flatten/Const?
flatten/ReshapeReshapeflatten/transpose:y:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten/Reshapes
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout/dropout/Const?
dropout/dropout/MulMulflatten/Reshape:output:0dropout/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/dropout/Mulq
dropout/dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
dropout/dropout/raten
dropout/dropout/seedConst*
_output_shapes
: *
dtype0*
value	B : 2
dropout/dropout/seed?
dropout/dropout
MLCDropoutflatten/Reshape:output:0dropout/dropout/rate:output:0dropout/dropout/seed:output:0*
T0*(
_output_shapes
:??????????2
dropout/dropout?
dense/MLCMatMul/ReadVariableOpReadVariableOp'dense_mlcmatmul_readvariableop_kernel_1*
_output_shapes
:	?2*
dtype02 
dense/MLCMatMul/ReadVariableOp?
dense/MLCMatMul	MLCMatMuldropout/dropout:output:0&dense/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
dense/MLCMatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp#dense_biasadd_readvariableop_bias_1*
_output_shapes
:2*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MLCMatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
dense/BiasAddj

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22

dense/Relu?
 dense_1/MLCMatMul/ReadVariableOpReadVariableOp)dense_1_mlcmatmul_readvariableop_kernel_2*
_output_shapes

:2*
dtype02"
 dense_1/MLCMatMul/ReadVariableOp?
dense_1/MLCMatMul	MLCMatMuldense/Relu:activations:0(dense_1/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/MLCMatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp%dense_1_biasadd_readvariableop_bias_2*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MLCMatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/BiasAddy
dense_1/SigmoidSigmoiddense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_1/Sigmoid?
IdentityIdentitydense_1/Sigmoid:y:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MLCMatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp!^dense_1/MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:??????????::::::2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/MLCMatMul/ReadVariableOpdense/MLCMatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2D
 dense_1/MLCMatMul/ReadVariableOp dense_1/MLCMatMul/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
F__inference_block3_conv2_layer_call_and_return_conditional_losses_2608

inputs-
)conv2d_readvariableop_block3_conv2_kernel,
(biasadd_readvariableop_block3_conv2_bias
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOp)conv2d_readvariableop_block3_conv2_kernel*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2D	MLCConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
data_formatNCHW*
num_args *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOp(biasadd_readvariableop_block3_conv2_bias*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
data_formatNCHW2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
A__inference_model_1_layer_call_and_return_conditional_losses_2329

inputs:
6block1_conv1_conv2d_readvariableop_block1_conv1_kernel9
5block1_conv1_biasadd_readvariableop_block1_conv1_bias:
6block1_conv2_conv2d_readvariableop_block1_conv2_kernel9
5block1_conv2_biasadd_readvariableop_block1_conv2_bias:
6block2_conv1_conv2d_readvariableop_block2_conv1_kernel9
5block2_conv1_biasadd_readvariableop_block2_conv1_bias:
6block2_conv2_conv2d_readvariableop_block2_conv2_kernel9
5block2_conv2_biasadd_readvariableop_block2_conv2_bias:
6block3_conv1_conv2d_readvariableop_block3_conv1_kernel9
5block3_conv1_biasadd_readvariableop_block3_conv1_bias:
6block3_conv2_conv2d_readvariableop_block3_conv2_kernel9
5block3_conv2_biasadd_readvariableop_block3_conv2_bias:
6block3_conv3_conv2d_readvariableop_block3_conv3_kernel9
5block3_conv3_biasadd_readvariableop_block3_conv3_bias:
6block4_conv1_conv2d_readvariableop_block4_conv1_kernel9
5block4_conv1_biasadd_readvariableop_block4_conv1_bias:
6block4_conv2_conv2d_readvariableop_block4_conv2_kernel9
5block4_conv2_biasadd_readvariableop_block4_conv2_bias:
6block4_conv3_conv2d_readvariableop_block4_conv3_kernel9
5block4_conv3_biasadd_readvariableop_block4_conv3_bias:
6block5_conv1_conv2d_readvariableop_block5_conv1_kernel9
5block5_conv1_biasadd_readvariableop_block5_conv1_bias2
.sequential_conv2d_conv2d_readvariableop_kernel1
-sequential_conv2d_biasadd_readvariableop_bias6
2sequential_dense_mlcmatmul_readvariableop_kernel_12
.sequential_dense_biasadd_readvariableop_bias_18
4sequential_dense_1_mlcmatmul_readvariableop_kernel_24
0sequential_dense_1_biasadd_readvariableop_bias_2
identity??#block1_conv1/BiasAdd/ReadVariableOp?"block1_conv1/Conv2D/ReadVariableOp?#block1_conv2/BiasAdd/ReadVariableOp?"block1_conv2/Conv2D/ReadVariableOp?#block2_conv1/BiasAdd/ReadVariableOp?"block2_conv1/Conv2D/ReadVariableOp?#block2_conv2/BiasAdd/ReadVariableOp?"block2_conv2/Conv2D/ReadVariableOp?#block3_conv1/BiasAdd/ReadVariableOp?"block3_conv1/Conv2D/ReadVariableOp?#block3_conv2/BiasAdd/ReadVariableOp?"block3_conv2/Conv2D/ReadVariableOp?#block3_conv3/BiasAdd/ReadVariableOp?"block3_conv3/Conv2D/ReadVariableOp?#block4_conv1/BiasAdd/ReadVariableOp?"block4_conv1/Conv2D/ReadVariableOp?#block4_conv2/BiasAdd/ReadVariableOp?"block4_conv2/Conv2D/ReadVariableOp?#block4_conv3/BiasAdd/ReadVariableOp?"block4_conv3/Conv2D/ReadVariableOp?#block5_conv1/BiasAdd/ReadVariableOp?"block5_conv1/Conv2D/ReadVariableOp?(sequential/conv2d/BiasAdd/ReadVariableOp?'sequential/conv2d/Conv2D/ReadVariableOp?'sequential/dense/BiasAdd/ReadVariableOp?)sequential/dense/MLCMatMul/ReadVariableOp?)sequential/dense_1/BiasAdd/ReadVariableOp?+sequential/dense_1/MLCMatMul/ReadVariableOp?
"block1_conv1/Conv2D/ReadVariableOpReadVariableOp6block1_conv1_conv2d_readvariableop_block1_conv1_kernel*&
_output_shapes
:@*
dtype02$
"block1_conv1/Conv2D/ReadVariableOp?
block1_conv1/Conv2D	MLCConv2Dinputs*block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@xx*
data_formatNCHW*
num_args *
paddingSAME*
strides
2
block1_conv1/Conv2D?
#block1_conv1/BiasAdd/ReadVariableOpReadVariableOp5block1_conv1_biasadd_readvariableop_block1_conv1_bias*
_output_shapes
:@*
dtype02%
#block1_conv1/BiasAdd/ReadVariableOp?
block1_conv1/BiasAddBiasAddblock1_conv1/Conv2D:output:0+block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@xx*
data_formatNCHW2
block1_conv1/BiasAdd?
block1_conv1/ReluRelublock1_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@xx2
block1_conv1/Relu?
"block1_conv2/Conv2D/ReadVariableOpReadVariableOp6block1_conv2_conv2d_readvariableop_block1_conv2_kernel*&
_output_shapes
:@@*
dtype02$
"block1_conv2/Conv2D/ReadVariableOp?
block1_conv2/Conv2D	MLCConv2Dblock1_conv1/Relu:activations:0*block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@xx*
data_formatNCHW*
num_args *
paddingSAME*
strides
2
block1_conv2/Conv2D?
#block1_conv2/BiasAdd/ReadVariableOpReadVariableOp5block1_conv2_biasadd_readvariableop_block1_conv2_bias*
_output_shapes
:@*
dtype02%
#block1_conv2/BiasAdd/ReadVariableOp?
block1_conv2/BiasAddBiasAddblock1_conv2/Conv2D:output:0+block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@xx*
data_formatNCHW2
block1_conv2/BiasAdd?
block1_conv2/ReluRelublock1_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@xx2
block1_conv2/Relu?
block1_pool/MaxPoolMaxPoolblock1_conv2/Relu:activations:0*/
_output_shapes
:?????????@<<*
data_formatNCHW*
ksize
*
paddingVALID*
strides
2
block1_pool/MaxPool?
"block2_conv1/Conv2D/ReadVariableOpReadVariableOp6block2_conv1_conv2d_readvariableop_block2_conv1_kernel*'
_output_shapes
:@?*
dtype02$
"block2_conv1/Conv2D/ReadVariableOp?
block2_conv1/Conv2D	MLCConv2Dblock1_pool/MaxPool:output:0*block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????<<*
data_formatNCHW*
num_args *
paddingSAME*
strides
2
block2_conv1/Conv2D?
#block2_conv1/BiasAdd/ReadVariableOpReadVariableOp5block2_conv1_biasadd_readvariableop_block2_conv1_bias*
_output_shapes	
:?*
dtype02%
#block2_conv1/BiasAdd/ReadVariableOp?
block2_conv1/BiasAddBiasAddblock2_conv1/Conv2D:output:0+block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????<<*
data_formatNCHW2
block2_conv1/BiasAdd?
block2_conv1/ReluRelublock2_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:??????????<<2
block2_conv1/Relu?
"block2_conv2/Conv2D/ReadVariableOpReadVariableOp6block2_conv2_conv2d_readvariableop_block2_conv2_kernel*(
_output_shapes
:??*
dtype02$
"block2_conv2/Conv2D/ReadVariableOp?
block2_conv2/Conv2D	MLCConv2Dblock2_conv1/Relu:activations:0*block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????<<*
data_formatNCHW*
num_args *
paddingSAME*
strides
2
block2_conv2/Conv2D?
#block2_conv2/BiasAdd/ReadVariableOpReadVariableOp5block2_conv2_biasadd_readvariableop_block2_conv2_bias*
_output_shapes	
:?*
dtype02%
#block2_conv2/BiasAdd/ReadVariableOp?
block2_conv2/BiasAddBiasAddblock2_conv2/Conv2D:output:0+block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????<<*
data_formatNCHW2
block2_conv2/BiasAdd?
block2_conv2/ReluRelublock2_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:??????????<<2
block2_conv2/Relu?
block2_pool/MaxPoolMaxPoolblock2_conv2/Relu:activations:0*0
_output_shapes
:??????????*
data_formatNCHW*
ksize
*
paddingVALID*
strides
2
block2_pool/MaxPool?
"block3_conv1/Conv2D/ReadVariableOpReadVariableOp6block3_conv1_conv2d_readvariableop_block3_conv1_kernel*(
_output_shapes
:??*
dtype02$
"block3_conv1/Conv2D/ReadVariableOp?
block3_conv1/Conv2D	MLCConv2Dblock2_pool/MaxPool:output:0*block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
data_formatNCHW*
num_args *
paddingSAME*
strides
2
block3_conv1/Conv2D?
#block3_conv1/BiasAdd/ReadVariableOpReadVariableOp5block3_conv1_biasadd_readvariableop_block3_conv1_bias*
_output_shapes	
:?*
dtype02%
#block3_conv1/BiasAdd/ReadVariableOp?
block3_conv1/BiasAddBiasAddblock3_conv1/Conv2D:output:0+block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
data_formatNCHW2
block3_conv1/BiasAdd?
block3_conv1/ReluRelublock3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
block3_conv1/Relu?
"block3_conv2/Conv2D/ReadVariableOpReadVariableOp6block3_conv2_conv2d_readvariableop_block3_conv2_kernel*(
_output_shapes
:??*
dtype02$
"block3_conv2/Conv2D/ReadVariableOp?
block3_conv2/Conv2D	MLCConv2Dblock3_conv1/Relu:activations:0*block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
data_formatNCHW*
num_args *
paddingSAME*
strides
2
block3_conv2/Conv2D?
#block3_conv2/BiasAdd/ReadVariableOpReadVariableOp5block3_conv2_biasadd_readvariableop_block3_conv2_bias*
_output_shapes	
:?*
dtype02%
#block3_conv2/BiasAdd/ReadVariableOp?
block3_conv2/BiasAddBiasAddblock3_conv2/Conv2D:output:0+block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
data_formatNCHW2
block3_conv2/BiasAdd?
block3_conv2/ReluRelublock3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
block3_conv2/Relu?
"block3_conv3/Conv2D/ReadVariableOpReadVariableOp6block3_conv3_conv2d_readvariableop_block3_conv3_kernel*(
_output_shapes
:??*
dtype02$
"block3_conv3/Conv2D/ReadVariableOp?
block3_conv3/Conv2D	MLCConv2Dblock3_conv2/Relu:activations:0*block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
data_formatNCHW*
num_args *
paddingSAME*
strides
2
block3_conv3/Conv2D?
#block3_conv3/BiasAdd/ReadVariableOpReadVariableOp5block3_conv3_biasadd_readvariableop_block3_conv3_bias*
_output_shapes	
:?*
dtype02%
#block3_conv3/BiasAdd/ReadVariableOp?
block3_conv3/BiasAddBiasAddblock3_conv3/Conv2D:output:0+block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
data_formatNCHW2
block3_conv3/BiasAdd?
block3_conv3/ReluRelublock3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
block3_conv3/Relu?
block3_pool/MaxPoolMaxPoolblock3_conv3/Relu:activations:0*0
_output_shapes
:??????????*
data_formatNCHW*
ksize
*
paddingVALID*
strides
2
block3_pool/MaxPool?
"block4_conv1/Conv2D/ReadVariableOpReadVariableOp6block4_conv1_conv2d_readvariableop_block4_conv1_kernel*(
_output_shapes
:??*
dtype02$
"block4_conv1/Conv2D/ReadVariableOp?
block4_conv1/Conv2D	MLCConv2Dblock3_pool/MaxPool:output:0*block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
data_formatNCHW*
num_args *
paddingSAME*
strides
2
block4_conv1/Conv2D?
#block4_conv1/BiasAdd/ReadVariableOpReadVariableOp5block4_conv1_biasadd_readvariableop_block4_conv1_bias*
_output_shapes	
:?*
dtype02%
#block4_conv1/BiasAdd/ReadVariableOp?
block4_conv1/BiasAddBiasAddblock4_conv1/Conv2D:output:0+block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
data_formatNCHW2
block4_conv1/BiasAdd?
block4_conv1/ReluRelublock4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
block4_conv1/Relu?
"block4_conv2/Conv2D/ReadVariableOpReadVariableOp6block4_conv2_conv2d_readvariableop_block4_conv2_kernel*(
_output_shapes
:??*
dtype02$
"block4_conv2/Conv2D/ReadVariableOp?
block4_conv2/Conv2D	MLCConv2Dblock4_conv1/Relu:activations:0*block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
data_formatNCHW*
num_args *
paddingSAME*
strides
2
block4_conv2/Conv2D?
#block4_conv2/BiasAdd/ReadVariableOpReadVariableOp5block4_conv2_biasadd_readvariableop_block4_conv2_bias*
_output_shapes	
:?*
dtype02%
#block4_conv2/BiasAdd/ReadVariableOp?
block4_conv2/BiasAddBiasAddblock4_conv2/Conv2D:output:0+block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
data_formatNCHW2
block4_conv2/BiasAdd?
block4_conv2/ReluRelublock4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
block4_conv2/Relu?
"block4_conv3/Conv2D/ReadVariableOpReadVariableOp6block4_conv3_conv2d_readvariableop_block4_conv3_kernel*(
_output_shapes
:??*
dtype02$
"block4_conv3/Conv2D/ReadVariableOp?
block4_conv3/Conv2D	MLCConv2Dblock4_conv2/Relu:activations:0*block4_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
data_formatNCHW*
num_args *
paddingSAME*
strides
2
block4_conv3/Conv2D?
#block4_conv3/BiasAdd/ReadVariableOpReadVariableOp5block4_conv3_biasadd_readvariableop_block4_conv3_bias*
_output_shapes	
:?*
dtype02%
#block4_conv3/BiasAdd/ReadVariableOp?
block4_conv3/BiasAddBiasAddblock4_conv3/Conv2D:output:0+block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
data_formatNCHW2
block4_conv3/BiasAdd?
block4_conv3/ReluRelublock4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
block4_conv3/Relu?
block4_pool/MaxPoolMaxPoolblock4_conv3/Relu:activations:0*0
_output_shapes
:??????????*
data_formatNCHW*
ksize
*
paddingVALID*
strides
2
block4_pool/MaxPool?
"block5_conv1/Conv2D/ReadVariableOpReadVariableOp6block5_conv1_conv2d_readvariableop_block5_conv1_kernel*(
_output_shapes
:??*
dtype02$
"block5_conv1/Conv2D/ReadVariableOp?
block5_conv1/Conv2D	MLCConv2Dblock4_pool/MaxPool:output:0*block5_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
data_formatNCHW*
num_args *
paddingSAME*
strides
2
block5_conv1/Conv2D?
#block5_conv1/BiasAdd/ReadVariableOpReadVariableOp5block5_conv1_biasadd_readvariableop_block5_conv1_bias*
_output_shapes	
:?*
dtype02%
#block5_conv1/BiasAdd/ReadVariableOp?
block5_conv1/BiasAddBiasAddblock5_conv1/Conv2D:output:0+block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
data_formatNCHW2
block5_conv1/BiasAdd?
block5_conv1/ReluRelublock5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
block5_conv1/Relu?
'sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp.sequential_conv2d_conv2d_readvariableop_kernel*'
_output_shapes
:?*
dtype02)
'sequential/conv2d/Conv2D/ReadVariableOp?
sequential/conv2d/Conv2D	MLCConv2Dblock5_conv1/Relu:activations:0/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
data_formatNCHW*
num_args *
paddingVALID*
strides
2
sequential/conv2d/Conv2D?
(sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp-sequential_conv2d_biasadd_readvariableop_bias*
_output_shapes
:*
dtype02*
(sequential/conv2d/BiasAdd/ReadVariableOp?
sequential/conv2d/BiasAddBiasAdd!sequential/conv2d/Conv2D:output:00sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
data_formatNCHW2
sequential/conv2d/BiasAdd?
sequential/conv2d/ReluRelu"sequential/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
sequential/conv2d/Relu?
 sequential/max_pooling2d/MaxPoolMaxPool$sequential/conv2d/Relu:activations:0*/
_output_shapes
:?????????*
data_formatNCHW*
ksize
*
paddingVALID*
strides
2"
 sequential/max_pooling2d/MaxPool?
!sequential/flatten/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!sequential/flatten/transpose/perm?
sequential/flatten/transpose	Transpose)sequential/max_pooling2d/MaxPool:output:0*sequential/flatten/transpose/perm:output:0*
T0*/
_output_shapes
:?????????2
sequential/flatten/transpose?
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????   2
sequential/flatten/Const?
sequential/flatten/ReshapeReshape sequential/flatten/transpose:y:0!sequential/flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
sequential/flatten/Reshape?
 sequential/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2"
 sequential/dropout/dropout/Const?
sequential/dropout/dropout/MulMul#sequential/flatten/Reshape:output:0)sequential/dropout/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2 
sequential/dropout/dropout/Mul?
sequential/dropout/dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *???>2!
sequential/dropout/dropout/rate?
sequential/dropout/dropout/seedConst*
_output_shapes
: *
dtype0*
value	B : 2!
sequential/dropout/dropout/seed?
sequential/dropout/dropout
MLCDropout#sequential/flatten/Reshape:output:0(sequential/dropout/dropout/rate:output:0(sequential/dropout/dropout/seed:output:0*
T0*(
_output_shapes
:??????????2
sequential/dropout/dropout?
)sequential/dense/MLCMatMul/ReadVariableOpReadVariableOp2sequential_dense_mlcmatmul_readvariableop_kernel_1*
_output_shapes
:	?2*
dtype02+
)sequential/dense/MLCMatMul/ReadVariableOp?
sequential/dense/MLCMatMul	MLCMatMul#sequential/dropout/dropout:output:01sequential/dense/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
sequential/dense/MLCMatMul?
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp.sequential_dense_biasadd_readvariableop_bias_1*
_output_shapes
:2*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOp?
sequential/dense/BiasAddBiasAdd$sequential/dense/MLCMatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
sequential/dense/BiasAdd?
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22
sequential/dense/Relu?
+sequential/dense_1/MLCMatMul/ReadVariableOpReadVariableOp4sequential_dense_1_mlcmatmul_readvariableop_kernel_2*
_output_shapes

:2*
dtype02-
+sequential/dense_1/MLCMatMul/ReadVariableOp?
sequential/dense_1/MLCMatMul	MLCMatMul#sequential/dense/Relu:activations:03sequential/dense_1/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential/dense_1/MLCMatMul?
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_1_biasadd_readvariableop_bias_2*
_output_shapes
:*
dtype02+
)sequential/dense_1/BiasAdd/ReadVariableOp?
sequential/dense_1/BiasAddBiasAdd&sequential/dense_1/MLCMatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential/dense_1/BiasAdd?
sequential/dense_1/SigmoidSigmoid#sequential/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
sequential/dense_1/Sigmoid?	
IdentityIdentitysequential/dense_1/Sigmoid:y:0$^block1_conv1/BiasAdd/ReadVariableOp#^block1_conv1/Conv2D/ReadVariableOp$^block1_conv2/BiasAdd/ReadVariableOp#^block1_conv2/Conv2D/ReadVariableOp$^block2_conv1/BiasAdd/ReadVariableOp#^block2_conv1/Conv2D/ReadVariableOp$^block2_conv2/BiasAdd/ReadVariableOp#^block2_conv2/Conv2D/ReadVariableOp$^block3_conv1/BiasAdd/ReadVariableOp#^block3_conv1/Conv2D/ReadVariableOp$^block3_conv2/BiasAdd/ReadVariableOp#^block3_conv2/Conv2D/ReadVariableOp$^block3_conv3/BiasAdd/ReadVariableOp#^block3_conv3/Conv2D/ReadVariableOp$^block4_conv1/BiasAdd/ReadVariableOp#^block4_conv1/Conv2D/ReadVariableOp$^block4_conv2/BiasAdd/ReadVariableOp#^block4_conv2/Conv2D/ReadVariableOp$^block4_conv3/BiasAdd/ReadVariableOp#^block4_conv3/Conv2D/ReadVariableOp$^block5_conv1/BiasAdd/ReadVariableOp#^block5_conv1/Conv2D/ReadVariableOp)^sequential/conv2d/BiasAdd/ReadVariableOp(^sequential/conv2d/Conv2D/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp*^sequential/dense/MLCMatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp,^sequential/dense_1/MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????xx::::::::::::::::::::::::::::2J
#block1_conv1/BiasAdd/ReadVariableOp#block1_conv1/BiasAdd/ReadVariableOp2H
"block1_conv1/Conv2D/ReadVariableOp"block1_conv1/Conv2D/ReadVariableOp2J
#block1_conv2/BiasAdd/ReadVariableOp#block1_conv2/BiasAdd/ReadVariableOp2H
"block1_conv2/Conv2D/ReadVariableOp"block1_conv2/Conv2D/ReadVariableOp2J
#block2_conv1/BiasAdd/ReadVariableOp#block2_conv1/BiasAdd/ReadVariableOp2H
"block2_conv1/Conv2D/ReadVariableOp"block2_conv1/Conv2D/ReadVariableOp2J
#block2_conv2/BiasAdd/ReadVariableOp#block2_conv2/BiasAdd/ReadVariableOp2H
"block2_conv2/Conv2D/ReadVariableOp"block2_conv2/Conv2D/ReadVariableOp2J
#block3_conv1/BiasAdd/ReadVariableOp#block3_conv1/BiasAdd/ReadVariableOp2H
"block3_conv1/Conv2D/ReadVariableOp"block3_conv1/Conv2D/ReadVariableOp2J
#block3_conv2/BiasAdd/ReadVariableOp#block3_conv2/BiasAdd/ReadVariableOp2H
"block3_conv2/Conv2D/ReadVariableOp"block3_conv2/Conv2D/ReadVariableOp2J
#block3_conv3/BiasAdd/ReadVariableOp#block3_conv3/BiasAdd/ReadVariableOp2H
"block3_conv3/Conv2D/ReadVariableOp"block3_conv3/Conv2D/ReadVariableOp2J
#block4_conv1/BiasAdd/ReadVariableOp#block4_conv1/BiasAdd/ReadVariableOp2H
"block4_conv1/Conv2D/ReadVariableOp"block4_conv1/Conv2D/ReadVariableOp2J
#block4_conv2/BiasAdd/ReadVariableOp#block4_conv2/BiasAdd/ReadVariableOp2H
"block4_conv2/Conv2D/ReadVariableOp"block4_conv2/Conv2D/ReadVariableOp2J
#block4_conv3/BiasAdd/ReadVariableOp#block4_conv3/BiasAdd/ReadVariableOp2H
"block4_conv3/Conv2D/ReadVariableOp"block4_conv3/Conv2D/ReadVariableOp2J
#block5_conv1/BiasAdd/ReadVariableOp#block5_conv1/BiasAdd/ReadVariableOp2H
"block5_conv1/Conv2D/ReadVariableOp"block5_conv1/Conv2D/ReadVariableOp2T
(sequential/conv2d/BiasAdd/ReadVariableOp(sequential/conv2d/BiasAdd/ReadVariableOp2R
'sequential/conv2d/Conv2D/ReadVariableOp'sequential/conv2d/Conv2D/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2V
)sequential/dense/MLCMatMul/ReadVariableOp)sequential/dense/MLCMatMul/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2Z
+sequential/dense_1/MLCMatMul/ReadVariableOp+sequential/dense_1/MLCMatMul/ReadVariableOp:W S
/
_output_shapes
:?????????xx
 
_user_specified_nameinputs
?)
?
D__inference_sequential_layer_call_and_return_conditional_losses_2740
inputs_0'
#conv2d_conv2d_readvariableop_kernel&
"conv2d_biasadd_readvariableop_bias+
'dense_mlcmatmul_readvariableop_kernel_1'
#dense_biasadd_readvariableop_bias_1-
)dense_1_mlcmatmul_readvariableop_kernel_2)
%dense_1_biasadd_readvariableop_bias_2
identity??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MLCMatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp? dense_1/MLCMatMul/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp#conv2d_conv2d_readvariableop_kernel*'
_output_shapes
:?*
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2D	MLCConv2Dinputs_0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
data_formatNCHW*
num_args *
paddingVALID*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp"conv2d_biasadd_readvariableop_bias*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
data_formatNCHW2
conv2d/BiasAddu
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d/Relu?
max_pooling2d/MaxPoolMaxPoolconv2d/Relu:activations:0*/
_output_shapes
:?????????*
data_formatNCHW*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool?
flatten/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
flatten/transpose/perm?
flatten/transpose	Transposemax_pooling2d/MaxPool:output:0flatten/transpose/perm:output:0*
T0*/
_output_shapes
:?????????2
flatten/transposeo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????   2
flatten/Const?
flatten/ReshapeReshapeflatten/transpose:y:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten/Reshapes
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout/dropout/Const?
dropout/dropout/MulMulflatten/Reshape:output:0dropout/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/dropout/Mulq
dropout/dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
dropout/dropout/raten
dropout/dropout/seedConst*
_output_shapes
: *
dtype0*
value	B : 2
dropout/dropout/seed?
dropout/dropout
MLCDropoutflatten/Reshape:output:0dropout/dropout/rate:output:0dropout/dropout/seed:output:0*
T0*(
_output_shapes
:??????????2
dropout/dropout?
dense/MLCMatMul/ReadVariableOpReadVariableOp'dense_mlcmatmul_readvariableop_kernel_1*
_output_shapes
:	?2*
dtype02 
dense/MLCMatMul/ReadVariableOp?
dense/MLCMatMul	MLCMatMuldropout/dropout:output:0&dense/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
dense/MLCMatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp#dense_biasadd_readvariableop_bias_1*
_output_shapes
:2*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MLCMatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
dense/BiasAddj

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22

dense/Relu?
 dense_1/MLCMatMul/ReadVariableOpReadVariableOp)dense_1_mlcmatmul_readvariableop_kernel_2*
_output_shapes

:2*
dtype02"
 dense_1/MLCMatMul/ReadVariableOp?
dense_1/MLCMatMul	MLCMatMuldense/Relu:activations:0(dense_1/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/MLCMatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp%dense_1_biasadd_readvariableop_bias_2*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MLCMatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/BiasAddy
dense_1/SigmoidSigmoiddense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_1/Sigmoid?
IdentityIdentitydense_1/Sigmoid:y:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MLCMatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp!^dense_1/MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:??????????::::::2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/MLCMatMul/ReadVariableOpdense/MLCMatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2D
 dense_1/MLCMatMul/ReadVariableOp dense_1/MLCMatMul/ReadVariableOp:Z V
0
_output_shapes
:??????????
"
_user_specified_name
inputs/0
?
a
E__inference_block3_pool_layer_call_and_return_conditional_losses_1350

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
data_formatNCHW*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
_
A__inference_dropout_layer_call_and_return_conditional_losses_2926

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
!__inference_signature_wrapper_434
input_2
conv2d_kernel
conv2d_bias
dense_kernel

dense_bias
dense_1_kernel
dense_1_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_2conv2d_kernelconv2d_biasdense_kernel
dense_biasdense_1_kerneldense_1_bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference__wrapped_model_4232
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:??????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:??????????
!
_user_specified_name	input_2
?
y
&__inference_dense_1_layer_call_fn_2972

inputs
kernel_2

bias_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputskernel_2bias_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_15062
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????2::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
?
F__inference_block1_conv1_layer_call_and_return_conditional_losses_1605

inputs-
)conv2d_readvariableop_block1_conv1_kernel,
(biasadd_readvariableop_block1_conv1_bias
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOp)conv2d_readvariableop_block1_conv1_kernel*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp?
Conv2D	MLCConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@xx*
data_formatNCHW*
num_args *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOp(biasadd_readvariableop_block1_conv1_bias*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@xx*
data_formatNCHW2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@xx2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@xx2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????xx::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????xx
 
_user_specified_nameinputs
?
?
D__inference_sequential_layer_call_and_return_conditional_losses_1535
input_2
conv2d_kernel
conv2d_bias
dense_kernel_1
dense_bias_1
dense_1_kernel_2
dense_1_bias_2
identity??conv2d/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_2conv2d_kernelconv2d_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_14102 
conv2d/StatefulPartitionedCall?
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_13922
max_pooling2d/PartitionedCall?
flatten/PartitionedCallPartitionedCall&max_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_14362
flatten/PartitionedCall?
dropout/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_14592
dropout/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_kernel_1dense_bias_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_14832
dense/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_kernel_2dense_1_bias_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_15062!
dense_1/StatefulPartitionedCall?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:??????????::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:Y U
0
_output_shapes
:??????????
!
_user_specified_name	input_2
?
H
,__inference_max_pooling2d_layer_call_fn_1395

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_13922
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
+__inference_block5_conv1_layer_call_fn_2705

inputs
block5_conv1_kernel
block5_conv1_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsblock5_conv1_kernelblock5_conv1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block5_conv1_layer_call_and_return_conditional_losses_18592
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
&__inference_model_1_layer_call_fn_2507

inputs
block1_conv1_kernel
block1_conv1_bias
block1_conv2_kernel
block1_conv2_bias
block2_conv1_kernel
block2_conv1_bias
block2_conv2_kernel
block2_conv2_bias
block3_conv1_kernel
block3_conv1_bias
block3_conv2_kernel
block3_conv2_bias
block3_conv3_kernel
block3_conv3_bias
block4_conv1_kernel
block4_conv1_bias
block4_conv2_kernel
block4_conv2_bias
block4_conv3_kernel
block4_conv3_bias
block5_conv1_kernel
block5_conv1_bias

kernel
bias
kernel_1

bias_1
kernel_2

bias_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsblock1_conv1_kernelblock1_conv1_biasblock1_conv2_kernelblock1_conv2_biasblock2_conv1_kernelblock2_conv1_biasblock2_conv2_kernelblock2_conv2_biasblock3_conv1_kernelblock3_conv1_biasblock3_conv2_kernelblock3_conv2_biasblock3_conv3_kernelblock3_conv3_biasblock4_conv1_kernelblock4_conv1_biasblock4_conv2_kernelblock4_conv2_biasblock4_conv3_kernelblock4_conv3_biasblock5_conv1_kernelblock5_conv1_biaskernelbiaskernel_1bias_1kernel_2bias_2*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*>
_read_only_resource_inputs 
	
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_model_1_layer_call_and_return_conditional_losses_21472
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????xx::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????xx
 
_user_specified_nameinputs
?
a
E__inference_block1_pool_layer_call_and_return_conditional_losses_1324

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
data_formatNCHW*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?T
?
A__inference_model_1_layer_call_and_return_conditional_losses_2147

inputs$
 block1_conv1_block1_conv1_kernel"
block1_conv1_block1_conv1_bias$
 block1_conv2_block1_conv2_kernel"
block1_conv2_block1_conv2_bias$
 block2_conv1_block2_conv1_kernel"
block2_conv1_block2_conv1_bias$
 block2_conv2_block2_conv2_kernel"
block2_conv2_block2_conv2_bias$
 block3_conv1_block3_conv1_kernel"
block3_conv1_block3_conv1_bias$
 block3_conv2_block3_conv2_kernel"
block3_conv2_block3_conv2_bias$
 block3_conv3_block3_conv3_kernel"
block3_conv3_block3_conv3_bias$
 block4_conv1_block4_conv1_kernel"
block4_conv1_block4_conv1_bias$
 block4_conv2_block4_conv2_kernel"
block4_conv2_block4_conv2_bias$
 block4_conv3_block4_conv3_kernel"
block4_conv3_block4_conv3_bias$
 block5_conv1_block5_conv1_kernel"
block5_conv1_block5_conv1_bias
sequential_kernel
sequential_bias
sequential_kernel_1
sequential_bias_1
sequential_kernel_2
sequential_bias_2
identity??$block1_conv1/StatefulPartitionedCall?$block1_conv2/StatefulPartitionedCall?$block2_conv1/StatefulPartitionedCall?$block2_conv2/StatefulPartitionedCall?$block3_conv1/StatefulPartitionedCall?$block3_conv2/StatefulPartitionedCall?$block3_conv3/StatefulPartitionedCall?$block4_conv1/StatefulPartitionedCall?$block4_conv2/StatefulPartitionedCall?$block4_conv3/StatefulPartitionedCall?$block5_conv1/StatefulPartitionedCall?"sequential/StatefulPartitionedCall?
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCallinputs block1_conv1_block1_conv1_kernelblock1_conv1_block1_conv1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@xx*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block1_conv1_layer_call_and_return_conditional_losses_16052&
$block1_conv1/StatefulPartitionedCall?
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0 block1_conv2_block1_conv2_kernelblock1_conv2_block1_conv2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@xx*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block1_conv2_layer_call_and_return_conditional_losses_16282&
$block1_conv2/StatefulPartitionedCall?
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@<<* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_block1_pool_layer_call_and_return_conditional_losses_13242
block1_pool/PartitionedCall?
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall$block1_pool/PartitionedCall:output:0 block2_conv1_block2_conv1_kernelblock2_conv1_block2_conv1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????<<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block2_conv1_layer_call_and_return_conditional_losses_16572&
$block2_conv1/StatefulPartitionedCall?
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0 block2_conv2_block2_conv2_kernelblock2_conv2_block2_conv2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????<<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block2_conv2_layer_call_and_return_conditional_losses_16802&
$block2_conv2/StatefulPartitionedCall?
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_block2_pool_layer_call_and_return_conditional_losses_13412
block2_pool/PartitionedCall?
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall$block2_pool/PartitionedCall:output:0 block3_conv1_block3_conv1_kernelblock3_conv1_block3_conv1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block3_conv1_layer_call_and_return_conditional_losses_17092&
$block3_conv1/StatefulPartitionedCall?
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0 block3_conv2_block3_conv2_kernelblock3_conv2_block3_conv2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block3_conv2_layer_call_and_return_conditional_losses_17322&
$block3_conv2/StatefulPartitionedCall?
$block3_conv3/StatefulPartitionedCallStatefulPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0 block3_conv3_block3_conv3_kernelblock3_conv3_block3_conv3_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block3_conv3_layer_call_and_return_conditional_losses_17552&
$block3_conv3/StatefulPartitionedCall?
block3_pool/PartitionedCallPartitionedCall-block3_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_block3_pool_layer_call_and_return_conditional_losses_13582
block3_pool/PartitionedCall?
$block4_conv1/StatefulPartitionedCallStatefulPartitionedCall$block3_pool/PartitionedCall:output:0 block4_conv1_block4_conv1_kernelblock4_conv1_block4_conv1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block4_conv1_layer_call_and_return_conditional_losses_17842&
$block4_conv1/StatefulPartitionedCall?
$block4_conv2/StatefulPartitionedCallStatefulPartitionedCall-block4_conv1/StatefulPartitionedCall:output:0 block4_conv2_block4_conv2_kernelblock4_conv2_block4_conv2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block4_conv2_layer_call_and_return_conditional_losses_18072&
$block4_conv2/StatefulPartitionedCall?
$block4_conv3/StatefulPartitionedCallStatefulPartitionedCall-block4_conv2/StatefulPartitionedCall:output:0 block4_conv3_block4_conv3_kernelblock4_conv3_block4_conv3_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block4_conv3_layer_call_and_return_conditional_losses_18302&
$block4_conv3/StatefulPartitionedCall?
block4_pool/PartitionedCallPartitionedCall-block4_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_block4_pool_layer_call_and_return_conditional_losses_13752
block4_pool/PartitionedCall?
$block5_conv1/StatefulPartitionedCallStatefulPartitionedCall$block4_pool/PartitionedCall:output:0 block5_conv1_block5_conv1_kernelblock5_conv1_block5_conv1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block5_conv1_layer_call_and_return_conditional_losses_18592&
$block5_conv1/StatefulPartitionedCall?
"sequential/StatefulPartitionedCallStatefulPartitionedCall-block5_conv1/StatefulPartitionedCall:output:0sequential_kernelsequential_biassequential_kernel_1sequential_bias_1sequential_kernel_2sequential_bias_2*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_19362$
"sequential/StatefulPartitionedCall?
IdentityIdentity+sequential/StatefulPartitionedCall:output:0%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall#^sequential/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????xx::::::::::::::::::::::::::::2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall2L
$block3_conv2/StatefulPartitionedCall$block3_conv2/StatefulPartitionedCall2L
$block3_conv3/StatefulPartitionedCall$block3_conv3/StatefulPartitionedCall2L
$block4_conv1/StatefulPartitionedCall$block4_conv1/StatefulPartitionedCall2L
$block4_conv2/StatefulPartitionedCall$block4_conv2/StatefulPartitionedCall2L
$block4_conv3/StatefulPartitionedCall$block4_conv3/StatefulPartitionedCall2L
$block5_conv1/StatefulPartitionedCall$block5_conv1/StatefulPartitionedCall2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall:W S
/
_output_shapes
:?????????xx
 
_user_specified_nameinputs
?
?
)__inference_sequential_layer_call_fn_2793
inputs_0

kernel
bias
kernel_1

bias_1
kernel_2

bias_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0kernelbiaskernel_1bias_1kernel_2bias_2*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_19362
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:??????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
0
_output_shapes
:??????????
"
_user_specified_name
inputs/0
?
]
A__inference_dropout_layer_call_and_return_conditional_losses_2921

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/Mula
dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
dropout/rate^
dropout/seedConst*
_output_shapes
: *
dtype0*
value	B : 2
dropout/seed?
dropout
MLCDropoutinputsdropout/rate:output:0dropout/seed:output:0*
T0*(
_output_shapes
:??????????2	
dropoute
IdentityIdentitydropout:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
F__inference_block4_conv1_layer_call_and_return_conditional_losses_1784

inputs-
)conv2d_readvariableop_block4_conv1_kernel,
(biasadd_readvariableop_block4_conv1_bias
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOp)conv2d_readvariableop_block4_conv1_kernel*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2D	MLCConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
data_formatNCHW*
num_args *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOp(biasadd_readvariableop_block4_conv1_bias*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
data_formatNCHW2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
+__inference_block3_conv3_layer_call_fn_2633

inputs
block3_conv3_kernel
block3_conv3_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsblock3_conv3_kernelblock3_conv3_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block3_conv3_layer_call_and_return_conditional_losses_17552
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?T
?
A__inference_model_1_layer_call_and_return_conditional_losses_2015
input_1$
 block1_conv1_block1_conv1_kernel"
block1_conv1_block1_conv1_bias$
 block1_conv2_block1_conv2_kernel"
block1_conv2_block1_conv2_bias$
 block2_conv1_block2_conv1_kernel"
block2_conv1_block2_conv1_bias$
 block2_conv2_block2_conv2_kernel"
block2_conv2_block2_conv2_bias$
 block3_conv1_block3_conv1_kernel"
block3_conv1_block3_conv1_bias$
 block3_conv2_block3_conv2_kernel"
block3_conv2_block3_conv2_bias$
 block3_conv3_block3_conv3_kernel"
block3_conv3_block3_conv3_bias$
 block4_conv1_block4_conv1_kernel"
block4_conv1_block4_conv1_bias$
 block4_conv2_block4_conv2_kernel"
block4_conv2_block4_conv2_bias$
 block4_conv3_block4_conv3_kernel"
block4_conv3_block4_conv3_bias$
 block5_conv1_block5_conv1_kernel"
block5_conv1_block5_conv1_bias
sequential_kernel
sequential_bias
sequential_kernel_1
sequential_bias_1
sequential_kernel_2
sequential_bias_2
identity??$block1_conv1/StatefulPartitionedCall?$block1_conv2/StatefulPartitionedCall?$block2_conv1/StatefulPartitionedCall?$block2_conv2/StatefulPartitionedCall?$block3_conv1/StatefulPartitionedCall?$block3_conv2/StatefulPartitionedCall?$block3_conv3/StatefulPartitionedCall?$block4_conv1/StatefulPartitionedCall?$block4_conv2/StatefulPartitionedCall?$block4_conv3/StatefulPartitionedCall?$block5_conv1/StatefulPartitionedCall?"sequential/StatefulPartitionedCall?
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCallinput_1 block1_conv1_block1_conv1_kernelblock1_conv1_block1_conv1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@xx*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block1_conv1_layer_call_and_return_conditional_losses_16052&
$block1_conv1/StatefulPartitionedCall?
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0 block1_conv2_block1_conv2_kernelblock1_conv2_block1_conv2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@xx*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block1_conv2_layer_call_and_return_conditional_losses_16282&
$block1_conv2/StatefulPartitionedCall?
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@<<* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_block1_pool_layer_call_and_return_conditional_losses_13242
block1_pool/PartitionedCall?
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall$block1_pool/PartitionedCall:output:0 block2_conv1_block2_conv1_kernelblock2_conv1_block2_conv1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????<<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block2_conv1_layer_call_and_return_conditional_losses_16572&
$block2_conv1/StatefulPartitionedCall?
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0 block2_conv2_block2_conv2_kernelblock2_conv2_block2_conv2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????<<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block2_conv2_layer_call_and_return_conditional_losses_16802&
$block2_conv2/StatefulPartitionedCall?
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_block2_pool_layer_call_and_return_conditional_losses_13412
block2_pool/PartitionedCall?
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall$block2_pool/PartitionedCall:output:0 block3_conv1_block3_conv1_kernelblock3_conv1_block3_conv1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block3_conv1_layer_call_and_return_conditional_losses_17092&
$block3_conv1/StatefulPartitionedCall?
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0 block3_conv2_block3_conv2_kernelblock3_conv2_block3_conv2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block3_conv2_layer_call_and_return_conditional_losses_17322&
$block3_conv2/StatefulPartitionedCall?
$block3_conv3/StatefulPartitionedCallStatefulPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0 block3_conv3_block3_conv3_kernelblock3_conv3_block3_conv3_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block3_conv3_layer_call_and_return_conditional_losses_17552&
$block3_conv3/StatefulPartitionedCall?
block3_pool/PartitionedCallPartitionedCall-block3_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_block3_pool_layer_call_and_return_conditional_losses_13582
block3_pool/PartitionedCall?
$block4_conv1/StatefulPartitionedCallStatefulPartitionedCall$block3_pool/PartitionedCall:output:0 block4_conv1_block4_conv1_kernelblock4_conv1_block4_conv1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block4_conv1_layer_call_and_return_conditional_losses_17842&
$block4_conv1/StatefulPartitionedCall?
$block4_conv2/StatefulPartitionedCallStatefulPartitionedCall-block4_conv1/StatefulPartitionedCall:output:0 block4_conv2_block4_conv2_kernelblock4_conv2_block4_conv2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block4_conv2_layer_call_and_return_conditional_losses_18072&
$block4_conv2/StatefulPartitionedCall?
$block4_conv3/StatefulPartitionedCallStatefulPartitionedCall-block4_conv2/StatefulPartitionedCall:output:0 block4_conv3_block4_conv3_kernelblock4_conv3_block4_conv3_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block4_conv3_layer_call_and_return_conditional_losses_18302&
$block4_conv3/StatefulPartitionedCall?
block4_pool/PartitionedCallPartitionedCall-block4_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_block4_pool_layer_call_and_return_conditional_losses_13752
block4_pool/PartitionedCall?
$block5_conv1/StatefulPartitionedCallStatefulPartitionedCall$block4_pool/PartitionedCall:output:0 block5_conv1_block5_conv1_kernelblock5_conv1_block5_conv1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block5_conv1_layer_call_and_return_conditional_losses_18592&
$block5_conv1/StatefulPartitionedCall?
"sequential/StatefulPartitionedCallStatefulPartitionedCall-block5_conv1/StatefulPartitionedCall:output:0sequential_kernelsequential_biassequential_kernel_1sequential_bias_1sequential_kernel_2sequential_bias_2*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_19362$
"sequential/StatefulPartitionedCall?
IdentityIdentity+sequential/StatefulPartitionedCall:output:0%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall#^sequential/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????xx::::::::::::::::::::::::::::2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall2L
$block3_conv2/StatefulPartitionedCall$block3_conv2/StatefulPartitionedCall2L
$block3_conv3/StatefulPartitionedCall$block3_conv3/StatefulPartitionedCall2L
$block4_conv1/StatefulPartitionedCall$block4_conv1/StatefulPartitionedCall2L
$block4_conv2/StatefulPartitionedCall$block4_conv2/StatefulPartitionedCall2L
$block4_conv3/StatefulPartitionedCall$block4_conv3/StatefulPartitionedCall2L
$block5_conv1/StatefulPartitionedCall$block5_conv1/StatefulPartitionedCall2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall:X T
/
_output_shapes
:?????????xx
!
_user_specified_name	input_1
?
?
+__inference_block1_conv2_layer_call_fn_2543

inputs
block1_conv2_kernel
block1_conv2_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsblock1_conv2_kernelblock1_conv2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@xx*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block1_conv2_layer_call_and_return_conditional_losses_16282
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????@xx2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@xx::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@xx
 
_user_specified_nameinputs
?
?
F__inference_block4_conv3_layer_call_and_return_conditional_losses_2680

inputs-
)conv2d_readvariableop_block4_conv3_kernel,
(biasadd_readvariableop_block4_conv3_bias
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOp)conv2d_readvariableop_block4_conv3_kernel*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2D	MLCConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
data_formatNCHW*
num_args *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOp(biasadd_readvariableop_block4_conv3_bias*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
data_formatNCHW2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?)
?
D__inference_sequential_layer_call_and_return_conditional_losses_2828

inputs'
#conv2d_conv2d_readvariableop_kernel&
"conv2d_biasadd_readvariableop_bias+
'dense_mlcmatmul_readvariableop_kernel_1'
#dense_biasadd_readvariableop_bias_1-
)dense_1_mlcmatmul_readvariableop_kernel_2)
%dense_1_biasadd_readvariableop_bias_2
identity??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MLCMatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp? dense_1/MLCMatMul/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp#conv2d_conv2d_readvariableop_kernel*'
_output_shapes
:?*
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2D	MLCConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
data_formatNCHW*
num_args *
paddingVALID*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp"conv2d_biasadd_readvariableop_bias*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
data_formatNCHW2
conv2d/BiasAddu
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d/Relu?
max_pooling2d/MaxPoolMaxPoolconv2d/Relu:activations:0*/
_output_shapes
:?????????*
data_formatNCHW*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool?
flatten/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
flatten/transpose/perm?
flatten/transpose	Transposemax_pooling2d/MaxPool:output:0flatten/transpose/perm:output:0*
T0*/
_output_shapes
:?????????2
flatten/transposeo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????   2
flatten/Const?
flatten/ReshapeReshapeflatten/transpose:y:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten/Reshapes
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?2
dropout/dropout/Const?
dropout/dropout/MulMulflatten/Reshape:output:0dropout/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/dropout/Mulq
dropout/dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *???>2
dropout/dropout/raten
dropout/dropout/seedConst*
_output_shapes
: *
dtype0*
value	B : 2
dropout/dropout/seed?
dropout/dropout
MLCDropoutflatten/Reshape:output:0dropout/dropout/rate:output:0dropout/dropout/seed:output:0*
T0*(
_output_shapes
:??????????2
dropout/dropout?
dense/MLCMatMul/ReadVariableOpReadVariableOp'dense_mlcmatmul_readvariableop_kernel_1*
_output_shapes
:	?2*
dtype02 
dense/MLCMatMul/ReadVariableOp?
dense/MLCMatMul	MLCMatMuldropout/dropout:output:0&dense/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
dense/MLCMatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp#dense_biasadd_readvariableop_bias_1*
_output_shapes
:2*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MLCMatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
dense/BiasAddj

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22

dense/Relu?
 dense_1/MLCMatMul/ReadVariableOpReadVariableOp)dense_1_mlcmatmul_readvariableop_kernel_2*
_output_shapes

:2*
dtype02"
 dense_1/MLCMatMul/ReadVariableOp?
dense_1/MLCMatMul	MLCMatMuldense/Relu:activations:0(dense_1/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/MLCMatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp%dense_1_biasadd_readvariableop_bias_2*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MLCMatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/BiasAddy
dense_1/SigmoidSigmoiddense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_1/Sigmoid?
IdentityIdentitydense_1/Sigmoid:y:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MLCMatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp!^dense_1/MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:??????????::::::2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/MLCMatMul/ReadVariableOpdense/MLCMatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2D
 dense_1/MLCMatMul/ReadVariableOp dense_1/MLCMatMul/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
F__inference_block4_conv1_layer_call_and_return_conditional_losses_2644

inputs-
)conv2d_readvariableop_block4_conv1_kernel,
(biasadd_readvariableop_block4_conv1_bias
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOp)conv2d_readvariableop_block4_conv1_kernel*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2D	MLCConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
data_formatNCHW*
num_args *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOp(biasadd_readvariableop_block4_conv1_bias*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
data_formatNCHW2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
a
E__inference_block2_pool_layer_call_and_return_conditional_losses_1341

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
data_formatNCHW*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
_
A__inference_dropout_layer_call_and_return_conditional_losses_1459

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
F
*__inference_block3_pool_layer_call_fn_1361

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_block3_pool_layer_call_and_return_conditional_losses_13582
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
"__inference_signature_wrapper_2213
input_1
block1_conv1_kernel
block1_conv1_bias
block1_conv2_kernel
block1_conv2_bias
block2_conv1_kernel
block2_conv1_bias
block2_conv2_kernel
block2_conv2_bias
block3_conv1_kernel
block3_conv1_bias
block3_conv2_kernel
block3_conv2_bias
block3_conv3_kernel
block3_conv3_bias
block4_conv1_kernel
block4_conv1_bias
block4_conv2_kernel
block4_conv2_bias
block4_conv3_kernel
block4_conv3_bias
block5_conv1_kernel
block5_conv1_bias

kernel
bias
kernel_1

bias_1
kernel_2

bias_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1block1_conv1_kernelblock1_conv1_biasblock1_conv2_kernelblock1_conv2_biasblock2_conv1_kernelblock2_conv1_biasblock2_conv2_kernelblock2_conv2_biasblock3_conv1_kernelblock3_conv1_biasblock3_conv2_kernelblock3_conv2_biasblock3_conv3_kernelblock3_conv3_biasblock4_conv1_kernelblock4_conv1_biasblock4_conv2_kernelblock4_conv2_biasblock4_conv3_kernelblock4_conv3_biasblock5_conv1_kernelblock5_conv1_biaskernelbiaskernel_1bias_1kernel_2bias_2*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*>
_read_only_resource_inputs 
	
*-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__wrapped_model_13102
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????xx::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????xx
!
_user_specified_name	input_1
?
?
F__inference_block3_conv1_layer_call_and_return_conditional_losses_2590

inputs-
)conv2d_readvariableop_block3_conv1_kernel,
(biasadd_readvariableop_block3_conv1_bias
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOp)conv2d_readvariableop_block3_conv1_kernel*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2D	MLCConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
data_formatNCHW*
num_args *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOp(biasadd_readvariableop_block3_conv1_bias*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
data_formatNCHW2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
F__inference_block2_conv2_layer_call_and_return_conditional_losses_1680

inputs-
)conv2d_readvariableop_block2_conv2_kernel,
(biasadd_readvariableop_block2_conv2_bias
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOp)conv2d_readvariableop_block2_conv2_kernel*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2D	MLCConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????<<*
data_formatNCHW*
num_args *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOp(biasadd_readvariableop_block2_conv2_bias*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????<<*
data_formatNCHW2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????<<2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????<<2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????<<::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????<<
 
_user_specified_nameinputs
?
a
E__inference_block2_pool_layer_call_and_return_conditional_losses_1333

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
data_formatNCHW*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
a
E__inference_block4_pool_layer_call_and_return_conditional_losses_1367

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
data_formatNCHW*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

?
?__inference_dense_layer_call_and_return_conditional_losses_2947

inputs%
!mlcmatmul_readvariableop_kernel_1!
biasadd_readvariableop_bias_1
identity??BiasAdd/ReadVariableOp?MLCMatMul/ReadVariableOp?
MLCMatMul/ReadVariableOpReadVariableOp!mlcmatmul_readvariableop_kernel_1*
_output_shapes
:	?2*
dtype02
MLCMatMul/ReadVariableOp
	MLCMatMul	MLCMatMulinputs MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
	MLCMatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_bias_1*
_output_shapes
:2*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMLCMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????22
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
MLCMatMul/ReadVariableOpMLCMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
+__inference_block4_conv2_layer_call_fn_2669

inputs
block4_conv2_kernel
block4_conv2_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsblock4_conv2_kernelblock4_conv2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block4_conv2_layer_call_and_return_conditional_losses_18072
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
F__inference_block2_conv1_layer_call_and_return_conditional_losses_2554

inputs-
)conv2d_readvariableop_block2_conv1_kernel,
(biasadd_readvariableop_block2_conv1_bias
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOp)conv2d_readvariableop_block2_conv1_kernel*'
_output_shapes
:@?*
dtype02
Conv2D/ReadVariableOp?
Conv2D	MLCConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????<<*
data_formatNCHW*
num_args *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOp(biasadd_readvariableop_block2_conv1_bias*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????<<*
data_formatNCHW2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????<<2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????<<2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@<<::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@<<
 
_user_specified_nameinputs
??
?
A__inference_model_1_layer_call_and_return_conditional_losses_2441

inputs:
6block1_conv1_conv2d_readvariableop_block1_conv1_kernel9
5block1_conv1_biasadd_readvariableop_block1_conv1_bias:
6block1_conv2_conv2d_readvariableop_block1_conv2_kernel9
5block1_conv2_biasadd_readvariableop_block1_conv2_bias:
6block2_conv1_conv2d_readvariableop_block2_conv1_kernel9
5block2_conv1_biasadd_readvariableop_block2_conv1_bias:
6block2_conv2_conv2d_readvariableop_block2_conv2_kernel9
5block2_conv2_biasadd_readvariableop_block2_conv2_bias:
6block3_conv1_conv2d_readvariableop_block3_conv1_kernel9
5block3_conv1_biasadd_readvariableop_block3_conv1_bias:
6block3_conv2_conv2d_readvariableop_block3_conv2_kernel9
5block3_conv2_biasadd_readvariableop_block3_conv2_bias:
6block3_conv3_conv2d_readvariableop_block3_conv3_kernel9
5block3_conv3_biasadd_readvariableop_block3_conv3_bias:
6block4_conv1_conv2d_readvariableop_block4_conv1_kernel9
5block4_conv1_biasadd_readvariableop_block4_conv1_bias:
6block4_conv2_conv2d_readvariableop_block4_conv2_kernel9
5block4_conv2_biasadd_readvariableop_block4_conv2_bias:
6block4_conv3_conv2d_readvariableop_block4_conv3_kernel9
5block4_conv3_biasadd_readvariableop_block4_conv3_bias:
6block5_conv1_conv2d_readvariableop_block5_conv1_kernel9
5block5_conv1_biasadd_readvariableop_block5_conv1_bias2
.sequential_conv2d_conv2d_readvariableop_kernel1
-sequential_conv2d_biasadd_readvariableop_bias6
2sequential_dense_mlcmatmul_readvariableop_kernel_12
.sequential_dense_biasadd_readvariableop_bias_18
4sequential_dense_1_mlcmatmul_readvariableop_kernel_24
0sequential_dense_1_biasadd_readvariableop_bias_2
identity??#block1_conv1/BiasAdd/ReadVariableOp?"block1_conv1/Conv2D/ReadVariableOp?#block1_conv2/BiasAdd/ReadVariableOp?"block1_conv2/Conv2D/ReadVariableOp?#block2_conv1/BiasAdd/ReadVariableOp?"block2_conv1/Conv2D/ReadVariableOp?#block2_conv2/BiasAdd/ReadVariableOp?"block2_conv2/Conv2D/ReadVariableOp?#block3_conv1/BiasAdd/ReadVariableOp?"block3_conv1/Conv2D/ReadVariableOp?#block3_conv2/BiasAdd/ReadVariableOp?"block3_conv2/Conv2D/ReadVariableOp?#block3_conv3/BiasAdd/ReadVariableOp?"block3_conv3/Conv2D/ReadVariableOp?#block4_conv1/BiasAdd/ReadVariableOp?"block4_conv1/Conv2D/ReadVariableOp?#block4_conv2/BiasAdd/ReadVariableOp?"block4_conv2/Conv2D/ReadVariableOp?#block4_conv3/BiasAdd/ReadVariableOp?"block4_conv3/Conv2D/ReadVariableOp?#block5_conv1/BiasAdd/ReadVariableOp?"block5_conv1/Conv2D/ReadVariableOp?(sequential/conv2d/BiasAdd/ReadVariableOp?'sequential/conv2d/Conv2D/ReadVariableOp?'sequential/dense/BiasAdd/ReadVariableOp?)sequential/dense/MLCMatMul/ReadVariableOp?)sequential/dense_1/BiasAdd/ReadVariableOp?+sequential/dense_1/MLCMatMul/ReadVariableOp?
"block1_conv1/Conv2D/ReadVariableOpReadVariableOp6block1_conv1_conv2d_readvariableop_block1_conv1_kernel*&
_output_shapes
:@*
dtype02$
"block1_conv1/Conv2D/ReadVariableOp?
block1_conv1/Conv2D	MLCConv2Dinputs*block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@xx*
data_formatNCHW*
num_args *
paddingSAME*
strides
2
block1_conv1/Conv2D?
#block1_conv1/BiasAdd/ReadVariableOpReadVariableOp5block1_conv1_biasadd_readvariableop_block1_conv1_bias*
_output_shapes
:@*
dtype02%
#block1_conv1/BiasAdd/ReadVariableOp?
block1_conv1/BiasAddBiasAddblock1_conv1/Conv2D:output:0+block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@xx*
data_formatNCHW2
block1_conv1/BiasAdd?
block1_conv1/ReluRelublock1_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@xx2
block1_conv1/Relu?
"block1_conv2/Conv2D/ReadVariableOpReadVariableOp6block1_conv2_conv2d_readvariableop_block1_conv2_kernel*&
_output_shapes
:@@*
dtype02$
"block1_conv2/Conv2D/ReadVariableOp?
block1_conv2/Conv2D	MLCConv2Dblock1_conv1/Relu:activations:0*block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@xx*
data_formatNCHW*
num_args *
paddingSAME*
strides
2
block1_conv2/Conv2D?
#block1_conv2/BiasAdd/ReadVariableOpReadVariableOp5block1_conv2_biasadd_readvariableop_block1_conv2_bias*
_output_shapes
:@*
dtype02%
#block1_conv2/BiasAdd/ReadVariableOp?
block1_conv2/BiasAddBiasAddblock1_conv2/Conv2D:output:0+block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@xx*
data_formatNCHW2
block1_conv2/BiasAdd?
block1_conv2/ReluRelublock1_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@xx2
block1_conv2/Relu?
block1_pool/MaxPoolMaxPoolblock1_conv2/Relu:activations:0*/
_output_shapes
:?????????@<<*
data_formatNCHW*
ksize
*
paddingVALID*
strides
2
block1_pool/MaxPool?
"block2_conv1/Conv2D/ReadVariableOpReadVariableOp6block2_conv1_conv2d_readvariableop_block2_conv1_kernel*'
_output_shapes
:@?*
dtype02$
"block2_conv1/Conv2D/ReadVariableOp?
block2_conv1/Conv2D	MLCConv2Dblock1_pool/MaxPool:output:0*block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????<<*
data_formatNCHW*
num_args *
paddingSAME*
strides
2
block2_conv1/Conv2D?
#block2_conv1/BiasAdd/ReadVariableOpReadVariableOp5block2_conv1_biasadd_readvariableop_block2_conv1_bias*
_output_shapes	
:?*
dtype02%
#block2_conv1/BiasAdd/ReadVariableOp?
block2_conv1/BiasAddBiasAddblock2_conv1/Conv2D:output:0+block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????<<*
data_formatNCHW2
block2_conv1/BiasAdd?
block2_conv1/ReluRelublock2_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:??????????<<2
block2_conv1/Relu?
"block2_conv2/Conv2D/ReadVariableOpReadVariableOp6block2_conv2_conv2d_readvariableop_block2_conv2_kernel*(
_output_shapes
:??*
dtype02$
"block2_conv2/Conv2D/ReadVariableOp?
block2_conv2/Conv2D	MLCConv2Dblock2_conv1/Relu:activations:0*block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????<<*
data_formatNCHW*
num_args *
paddingSAME*
strides
2
block2_conv2/Conv2D?
#block2_conv2/BiasAdd/ReadVariableOpReadVariableOp5block2_conv2_biasadd_readvariableop_block2_conv2_bias*
_output_shapes	
:?*
dtype02%
#block2_conv2/BiasAdd/ReadVariableOp?
block2_conv2/BiasAddBiasAddblock2_conv2/Conv2D:output:0+block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????<<*
data_formatNCHW2
block2_conv2/BiasAdd?
block2_conv2/ReluRelublock2_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:??????????<<2
block2_conv2/Relu?
block2_pool/MaxPoolMaxPoolblock2_conv2/Relu:activations:0*0
_output_shapes
:??????????*
data_formatNCHW*
ksize
*
paddingVALID*
strides
2
block2_pool/MaxPool?
"block3_conv1/Conv2D/ReadVariableOpReadVariableOp6block3_conv1_conv2d_readvariableop_block3_conv1_kernel*(
_output_shapes
:??*
dtype02$
"block3_conv1/Conv2D/ReadVariableOp?
block3_conv1/Conv2D	MLCConv2Dblock2_pool/MaxPool:output:0*block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
data_formatNCHW*
num_args *
paddingSAME*
strides
2
block3_conv1/Conv2D?
#block3_conv1/BiasAdd/ReadVariableOpReadVariableOp5block3_conv1_biasadd_readvariableop_block3_conv1_bias*
_output_shapes	
:?*
dtype02%
#block3_conv1/BiasAdd/ReadVariableOp?
block3_conv1/BiasAddBiasAddblock3_conv1/Conv2D:output:0+block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
data_formatNCHW2
block3_conv1/BiasAdd?
block3_conv1/ReluRelublock3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
block3_conv1/Relu?
"block3_conv2/Conv2D/ReadVariableOpReadVariableOp6block3_conv2_conv2d_readvariableop_block3_conv2_kernel*(
_output_shapes
:??*
dtype02$
"block3_conv2/Conv2D/ReadVariableOp?
block3_conv2/Conv2D	MLCConv2Dblock3_conv1/Relu:activations:0*block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
data_formatNCHW*
num_args *
paddingSAME*
strides
2
block3_conv2/Conv2D?
#block3_conv2/BiasAdd/ReadVariableOpReadVariableOp5block3_conv2_biasadd_readvariableop_block3_conv2_bias*
_output_shapes	
:?*
dtype02%
#block3_conv2/BiasAdd/ReadVariableOp?
block3_conv2/BiasAddBiasAddblock3_conv2/Conv2D:output:0+block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
data_formatNCHW2
block3_conv2/BiasAdd?
block3_conv2/ReluRelublock3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
block3_conv2/Relu?
"block3_conv3/Conv2D/ReadVariableOpReadVariableOp6block3_conv3_conv2d_readvariableop_block3_conv3_kernel*(
_output_shapes
:??*
dtype02$
"block3_conv3/Conv2D/ReadVariableOp?
block3_conv3/Conv2D	MLCConv2Dblock3_conv2/Relu:activations:0*block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
data_formatNCHW*
num_args *
paddingSAME*
strides
2
block3_conv3/Conv2D?
#block3_conv3/BiasAdd/ReadVariableOpReadVariableOp5block3_conv3_biasadd_readvariableop_block3_conv3_bias*
_output_shapes	
:?*
dtype02%
#block3_conv3/BiasAdd/ReadVariableOp?
block3_conv3/BiasAddBiasAddblock3_conv3/Conv2D:output:0+block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
data_formatNCHW2
block3_conv3/BiasAdd?
block3_conv3/ReluRelublock3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
block3_conv3/Relu?
block3_pool/MaxPoolMaxPoolblock3_conv3/Relu:activations:0*0
_output_shapes
:??????????*
data_formatNCHW*
ksize
*
paddingVALID*
strides
2
block3_pool/MaxPool?
"block4_conv1/Conv2D/ReadVariableOpReadVariableOp6block4_conv1_conv2d_readvariableop_block4_conv1_kernel*(
_output_shapes
:??*
dtype02$
"block4_conv1/Conv2D/ReadVariableOp?
block4_conv1/Conv2D	MLCConv2Dblock3_pool/MaxPool:output:0*block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
data_formatNCHW*
num_args *
paddingSAME*
strides
2
block4_conv1/Conv2D?
#block4_conv1/BiasAdd/ReadVariableOpReadVariableOp5block4_conv1_biasadd_readvariableop_block4_conv1_bias*
_output_shapes	
:?*
dtype02%
#block4_conv1/BiasAdd/ReadVariableOp?
block4_conv1/BiasAddBiasAddblock4_conv1/Conv2D:output:0+block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
data_formatNCHW2
block4_conv1/BiasAdd?
block4_conv1/ReluRelublock4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
block4_conv1/Relu?
"block4_conv2/Conv2D/ReadVariableOpReadVariableOp6block4_conv2_conv2d_readvariableop_block4_conv2_kernel*(
_output_shapes
:??*
dtype02$
"block4_conv2/Conv2D/ReadVariableOp?
block4_conv2/Conv2D	MLCConv2Dblock4_conv1/Relu:activations:0*block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
data_formatNCHW*
num_args *
paddingSAME*
strides
2
block4_conv2/Conv2D?
#block4_conv2/BiasAdd/ReadVariableOpReadVariableOp5block4_conv2_biasadd_readvariableop_block4_conv2_bias*
_output_shapes	
:?*
dtype02%
#block4_conv2/BiasAdd/ReadVariableOp?
block4_conv2/BiasAddBiasAddblock4_conv2/Conv2D:output:0+block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
data_formatNCHW2
block4_conv2/BiasAdd?
block4_conv2/ReluRelublock4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
block4_conv2/Relu?
"block4_conv3/Conv2D/ReadVariableOpReadVariableOp6block4_conv3_conv2d_readvariableop_block4_conv3_kernel*(
_output_shapes
:??*
dtype02$
"block4_conv3/Conv2D/ReadVariableOp?
block4_conv3/Conv2D	MLCConv2Dblock4_conv2/Relu:activations:0*block4_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
data_formatNCHW*
num_args *
paddingSAME*
strides
2
block4_conv3/Conv2D?
#block4_conv3/BiasAdd/ReadVariableOpReadVariableOp5block4_conv3_biasadd_readvariableop_block4_conv3_bias*
_output_shapes	
:?*
dtype02%
#block4_conv3/BiasAdd/ReadVariableOp?
block4_conv3/BiasAddBiasAddblock4_conv3/Conv2D:output:0+block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
data_formatNCHW2
block4_conv3/BiasAdd?
block4_conv3/ReluRelublock4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
block4_conv3/Relu?
block4_pool/MaxPoolMaxPoolblock4_conv3/Relu:activations:0*0
_output_shapes
:??????????*
data_formatNCHW*
ksize
*
paddingVALID*
strides
2
block4_pool/MaxPool?
"block5_conv1/Conv2D/ReadVariableOpReadVariableOp6block5_conv1_conv2d_readvariableop_block5_conv1_kernel*(
_output_shapes
:??*
dtype02$
"block5_conv1/Conv2D/ReadVariableOp?
block5_conv1/Conv2D	MLCConv2Dblock4_pool/MaxPool:output:0*block5_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
data_formatNCHW*
num_args *
paddingSAME*
strides
2
block5_conv1/Conv2D?
#block5_conv1/BiasAdd/ReadVariableOpReadVariableOp5block5_conv1_biasadd_readvariableop_block5_conv1_bias*
_output_shapes	
:?*
dtype02%
#block5_conv1/BiasAdd/ReadVariableOp?
block5_conv1/BiasAddBiasAddblock5_conv1/Conv2D:output:0+block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
data_formatNCHW2
block5_conv1/BiasAdd?
block5_conv1/ReluRelublock5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
block5_conv1/Relu?
'sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp.sequential_conv2d_conv2d_readvariableop_kernel*'
_output_shapes
:?*
dtype02)
'sequential/conv2d/Conv2D/ReadVariableOp?
sequential/conv2d/Conv2D	MLCConv2Dblock5_conv1/Relu:activations:0/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
data_formatNCHW*
num_args *
paddingVALID*
strides
2
sequential/conv2d/Conv2D?
(sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp-sequential_conv2d_biasadd_readvariableop_bias*
_output_shapes
:*
dtype02*
(sequential/conv2d/BiasAdd/ReadVariableOp?
sequential/conv2d/BiasAddBiasAdd!sequential/conv2d/Conv2D:output:00sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
data_formatNCHW2
sequential/conv2d/BiasAdd?
sequential/conv2d/ReluRelu"sequential/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
sequential/conv2d/Relu?
 sequential/max_pooling2d/MaxPoolMaxPool$sequential/conv2d/Relu:activations:0*/
_output_shapes
:?????????*
data_formatNCHW*
ksize
*
paddingVALID*
strides
2"
 sequential/max_pooling2d/MaxPool?
!sequential/flatten/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2#
!sequential/flatten/transpose/perm?
sequential/flatten/transpose	Transpose)sequential/max_pooling2d/MaxPool:output:0*sequential/flatten/transpose/perm:output:0*
T0*/
_output_shapes
:?????????2
sequential/flatten/transpose?
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????   2
sequential/flatten/Const?
sequential/flatten/ReshapeReshape sequential/flatten/transpose:y:0!sequential/flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
sequential/flatten/Reshape?
sequential/dropout/IdentityIdentity#sequential/flatten/Reshape:output:0*
T0*(
_output_shapes
:??????????2
sequential/dropout/Identity?
)sequential/dense/MLCMatMul/ReadVariableOpReadVariableOp2sequential_dense_mlcmatmul_readvariableop_kernel_1*
_output_shapes
:	?2*
dtype02+
)sequential/dense/MLCMatMul/ReadVariableOp?
sequential/dense/MLCMatMul	MLCMatMul$sequential/dropout/Identity:output:01sequential/dense/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
sequential/dense/MLCMatMul?
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp.sequential_dense_biasadd_readvariableop_bias_1*
_output_shapes
:2*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOp?
sequential/dense/BiasAddBiasAdd$sequential/dense/MLCMatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22
sequential/dense/BiasAdd?
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22
sequential/dense/Relu?
+sequential/dense_1/MLCMatMul/ReadVariableOpReadVariableOp4sequential_dense_1_mlcmatmul_readvariableop_kernel_2*
_output_shapes

:2*
dtype02-
+sequential/dense_1/MLCMatMul/ReadVariableOp?
sequential/dense_1/MLCMatMul	MLCMatMul#sequential/dense/Relu:activations:03sequential/dense_1/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential/dense_1/MLCMatMul?
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_1_biasadd_readvariableop_bias_2*
_output_shapes
:*
dtype02+
)sequential/dense_1/BiasAdd/ReadVariableOp?
sequential/dense_1/BiasAddBiasAdd&sequential/dense_1/MLCMatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential/dense_1/BiasAdd?
sequential/dense_1/SigmoidSigmoid#sequential/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
sequential/dense_1/Sigmoid?	
IdentityIdentitysequential/dense_1/Sigmoid:y:0$^block1_conv1/BiasAdd/ReadVariableOp#^block1_conv1/Conv2D/ReadVariableOp$^block1_conv2/BiasAdd/ReadVariableOp#^block1_conv2/Conv2D/ReadVariableOp$^block2_conv1/BiasAdd/ReadVariableOp#^block2_conv1/Conv2D/ReadVariableOp$^block2_conv2/BiasAdd/ReadVariableOp#^block2_conv2/Conv2D/ReadVariableOp$^block3_conv1/BiasAdd/ReadVariableOp#^block3_conv1/Conv2D/ReadVariableOp$^block3_conv2/BiasAdd/ReadVariableOp#^block3_conv2/Conv2D/ReadVariableOp$^block3_conv3/BiasAdd/ReadVariableOp#^block3_conv3/Conv2D/ReadVariableOp$^block4_conv1/BiasAdd/ReadVariableOp#^block4_conv1/Conv2D/ReadVariableOp$^block4_conv2/BiasAdd/ReadVariableOp#^block4_conv2/Conv2D/ReadVariableOp$^block4_conv3/BiasAdd/ReadVariableOp#^block4_conv3/Conv2D/ReadVariableOp$^block5_conv1/BiasAdd/ReadVariableOp#^block5_conv1/Conv2D/ReadVariableOp)^sequential/conv2d/BiasAdd/ReadVariableOp(^sequential/conv2d/Conv2D/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp*^sequential/dense/MLCMatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp,^sequential/dense_1/MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????xx::::::::::::::::::::::::::::2J
#block1_conv1/BiasAdd/ReadVariableOp#block1_conv1/BiasAdd/ReadVariableOp2H
"block1_conv1/Conv2D/ReadVariableOp"block1_conv1/Conv2D/ReadVariableOp2J
#block1_conv2/BiasAdd/ReadVariableOp#block1_conv2/BiasAdd/ReadVariableOp2H
"block1_conv2/Conv2D/ReadVariableOp"block1_conv2/Conv2D/ReadVariableOp2J
#block2_conv1/BiasAdd/ReadVariableOp#block2_conv1/BiasAdd/ReadVariableOp2H
"block2_conv1/Conv2D/ReadVariableOp"block2_conv1/Conv2D/ReadVariableOp2J
#block2_conv2/BiasAdd/ReadVariableOp#block2_conv2/BiasAdd/ReadVariableOp2H
"block2_conv2/Conv2D/ReadVariableOp"block2_conv2/Conv2D/ReadVariableOp2J
#block3_conv1/BiasAdd/ReadVariableOp#block3_conv1/BiasAdd/ReadVariableOp2H
"block3_conv1/Conv2D/ReadVariableOp"block3_conv1/Conv2D/ReadVariableOp2J
#block3_conv2/BiasAdd/ReadVariableOp#block3_conv2/BiasAdd/ReadVariableOp2H
"block3_conv2/Conv2D/ReadVariableOp"block3_conv2/Conv2D/ReadVariableOp2J
#block3_conv3/BiasAdd/ReadVariableOp#block3_conv3/BiasAdd/ReadVariableOp2H
"block3_conv3/Conv2D/ReadVariableOp"block3_conv3/Conv2D/ReadVariableOp2J
#block4_conv1/BiasAdd/ReadVariableOp#block4_conv1/BiasAdd/ReadVariableOp2H
"block4_conv1/Conv2D/ReadVariableOp"block4_conv1/Conv2D/ReadVariableOp2J
#block4_conv2/BiasAdd/ReadVariableOp#block4_conv2/BiasAdd/ReadVariableOp2H
"block4_conv2/Conv2D/ReadVariableOp"block4_conv2/Conv2D/ReadVariableOp2J
#block4_conv3/BiasAdd/ReadVariableOp#block4_conv3/BiasAdd/ReadVariableOp2H
"block4_conv3/Conv2D/ReadVariableOp"block4_conv3/Conv2D/ReadVariableOp2J
#block5_conv1/BiasAdd/ReadVariableOp#block5_conv1/BiasAdd/ReadVariableOp2H
"block5_conv1/Conv2D/ReadVariableOp"block5_conv1/Conv2D/ReadVariableOp2T
(sequential/conv2d/BiasAdd/ReadVariableOp(sequential/conv2d/BiasAdd/ReadVariableOp2R
'sequential/conv2d/Conv2D/ReadVariableOp'sequential/conv2d/Conv2D/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2V
)sequential/dense/MLCMatMul/ReadVariableOp)sequential/dense/MLCMatMul/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2Z
+sequential/dense_1/MLCMatMul/ReadVariableOp+sequential/dense_1/MLCMatMul/ReadVariableOp:W S
/
_output_shapes
:?????????xx
 
_user_specified_nameinputs
?
c
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_1384

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
data_formatNCHW*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
+__inference_block4_conv3_layer_call_fn_2687

inputs
block4_conv3_kernel
block4_conv3_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsblock4_conv3_kernelblock4_conv3_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block4_conv3_layer_call_and_return_conditional_losses_18302
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
F__inference_block1_conv1_layer_call_and_return_conditional_losses_2518

inputs-
)conv2d_readvariableop_block1_conv1_kernel,
(biasadd_readvariableop_block1_conv1_bias
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOp)conv2d_readvariableop_block1_conv1_kernel*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp?
Conv2D	MLCConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@xx*
data_formatNCHW*
num_args *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOp(biasadd_readvariableop_block1_conv1_bias*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@xx*
data_formatNCHW2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@xx2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@xx2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????xx::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????xx
 
_user_specified_nameinputs
?
?
F__inference_block3_conv1_layer_call_and_return_conditional_losses_1709

inputs-
)conv2d_readvariableop_block3_conv1_kernel,
(biasadd_readvariableop_block3_conv1_bias
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOp)conv2d_readvariableop_block3_conv1_kernel*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2D	MLCConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
data_formatNCHW*
num_args *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOp(biasadd_readvariableop_block3_conv1_bias*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
data_formatNCHW2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
+__inference_block2_conv1_layer_call_fn_2561

inputs
block2_conv1_kernel
block2_conv1_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsblock2_conv1_kernelblock2_conv1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????<<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block2_conv1_layer_call_and_return_conditional_losses_16572
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????<<2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@<<::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@<<
 
_user_specified_nameinputs
?

?
A__inference_dense_1_layer_call_and_return_conditional_losses_2965

inputs%
!mlcmatmul_readvariableop_kernel_2!
biasadd_readvariableop_bias_2
identity??BiasAdd/ReadVariableOp?MLCMatMul/ReadVariableOp?
MLCMatMul/ReadVariableOpReadVariableOp!mlcmatmul_readvariableop_kernel_2*
_output_shapes

:2*
dtype02
MLCMatMul/ReadVariableOp
	MLCMatMul	MLCMatMulinputs MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
	MLCMatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_bias_2*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMLCMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????2::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
MLCMatMul/ReadVariableOpMLCMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
?
&__inference_model_1_layer_call_fn_2097
input_1
block1_conv1_kernel
block1_conv1_bias
block1_conv2_kernel
block1_conv2_bias
block2_conv1_kernel
block2_conv1_bias
block2_conv2_kernel
block2_conv2_bias
block3_conv1_kernel
block3_conv1_bias
block3_conv2_kernel
block3_conv2_bias
block3_conv3_kernel
block3_conv3_bias
block4_conv1_kernel
block4_conv1_bias
block4_conv2_kernel
block4_conv2_bias
block4_conv3_kernel
block4_conv3_bias
block5_conv1_kernel
block5_conv1_bias

kernel
bias
kernel_1

bias_1
kernel_2

bias_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1block1_conv1_kernelblock1_conv1_biasblock1_conv2_kernelblock1_conv2_biasblock2_conv1_kernelblock2_conv1_biasblock2_conv2_kernelblock2_conv2_biasblock3_conv1_kernelblock3_conv1_biasblock3_conv2_kernelblock3_conv2_biasblock3_conv3_kernelblock3_conv3_biasblock4_conv1_kernelblock4_conv1_biasblock4_conv2_kernelblock4_conv2_biasblock4_conv3_kernelblock4_conv3_biasblock5_conv1_kernelblock5_conv1_biaskernelbiaskernel_1bias_1kernel_2bias_2*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*>
_read_only_resource_inputs 
	
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_model_1_layer_call_and_return_conditional_losses_20662
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????xx::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????xx
!
_user_specified_name	input_1
?|
?
 __inference__traced_restore_3185
file_prefix(
$assignvariableop_block1_conv1_kernel(
$assignvariableop_1_block1_conv1_bias*
&assignvariableop_2_block1_conv2_kernel(
$assignvariableop_3_block1_conv2_bias*
&assignvariableop_4_block2_conv1_kernel(
$assignvariableop_5_block2_conv1_bias*
&assignvariableop_6_block2_conv2_kernel(
$assignvariableop_7_block2_conv2_bias*
&assignvariableop_8_block3_conv1_kernel(
$assignvariableop_9_block3_conv1_bias+
'assignvariableop_10_block3_conv2_kernel)
%assignvariableop_11_block3_conv2_bias+
'assignvariableop_12_block3_conv3_kernel)
%assignvariableop_13_block3_conv3_bias+
'assignvariableop_14_block4_conv1_kernel)
%assignvariableop_15_block4_conv1_bias+
'assignvariableop_16_block4_conv2_kernel)
%assignvariableop_17_block4_conv2_bias+
'assignvariableop_18_block4_conv3_kernel)
%assignvariableop_19_block4_conv3_bias+
'assignvariableop_20_block5_conv1_kernel)
%assignvariableop_21_block5_conv1_bias%
!assignvariableop_22_conv2d_kernel#
assignvariableop_23_conv2d_bias$
 assignvariableop_24_dense_kernel"
assignvariableop_25_dense_bias&
"assignvariableop_26_dense_1_kernel$
 assignvariableop_27_dense_1_bias
assignvariableop_28_total_1
assignvariableop_29_count_1
identity_31??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/22/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/23/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/24/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/25/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/26/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/27/.ATTRIBUTES/VARIABLE_VALUEBJlayer_with_weights-11/keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEBJlayer_with_weights-11/keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Q
valueHBFB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes~
|:::::::::::::::::::::::::::::::*-
dtypes#
!22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp$assignvariableop_block1_conv1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp$assignvariableop_1_block1_conv1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp&assignvariableop_2_block1_conv2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp$assignvariableop_3_block1_conv2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp&assignvariableop_4_block2_conv1_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp$assignvariableop_5_block2_conv1_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp&assignvariableop_6_block2_conv2_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp$assignvariableop_7_block2_conv2_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp&assignvariableop_8_block3_conv1_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp$assignvariableop_9_block3_conv1_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp'assignvariableop_10_block3_conv2_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp%assignvariableop_11_block3_conv2_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp'assignvariableop_12_block3_conv3_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp%assignvariableop_13_block3_conv3_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp'assignvariableop_14_block4_conv1_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp%assignvariableop_15_block4_conv1_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp'assignvariableop_16_block4_conv2_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp%assignvariableop_17_block4_conv2_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp'assignvariableop_18_block4_conv3_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp%assignvariableop_19_block4_conv3_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp'assignvariableop_20_block5_conv1_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp%assignvariableop_21_block5_conv1_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp!assignvariableop_22_conv2d_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOpassignvariableop_23_conv2d_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp assignvariableop_24_dense_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOpassignvariableop_25_dense_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp"assignvariableop_26_dense_1_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp assignvariableop_27_dense_1_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOpassignvariableop_28_total_1Identity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOpassignvariableop_29_count_1Identity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_299
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_30Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_30?
Identity_31IdentityIdentity_30:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_31"#
identity_31Identity_31:output:0*?
_input_shapes|
z: ::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
F__inference_block2_conv1_layer_call_and_return_conditional_losses_1657

inputs-
)conv2d_readvariableop_block2_conv1_kernel,
(biasadd_readvariableop_block2_conv1_bias
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOp)conv2d_readvariableop_block2_conv1_kernel*'
_output_shapes
:@?*
dtype02
Conv2D/ReadVariableOp?
Conv2D	MLCConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????<<*
data_formatNCHW*
num_args *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOp(biasadd_readvariableop_block2_conv1_bias*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????<<*
data_formatNCHW2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????<<2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????<<2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@<<::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@<<
 
_user_specified_nameinputs
?
B
&__inference_flatten_layer_call_fn_2912

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_14362
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
F__inference_block5_conv1_layer_call_and_return_conditional_losses_1859

inputs-
)conv2d_readvariableop_block5_conv1_kernel,
(biasadd_readvariableop_block5_conv1_bias
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOp)conv2d_readvariableop_block5_conv1_kernel*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2D	MLCConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
data_formatNCHW*
num_args *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOp(biasadd_readvariableop_block5_conv1_bias*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
data_formatNCHW2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
+__inference_block4_conv1_layer_call_fn_2651

inputs
block4_conv1_kernel
block4_conv1_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsblock4_conv1_kernelblock4_conv1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block4_conv1_layer_call_and_return_conditional_losses_17842
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?T
?
A__inference_model_1_layer_call_and_return_conditional_losses_2066

inputs$
 block1_conv1_block1_conv1_kernel"
block1_conv1_block1_conv1_bias$
 block1_conv2_block1_conv2_kernel"
block1_conv2_block1_conv2_bias$
 block2_conv1_block2_conv1_kernel"
block2_conv1_block2_conv1_bias$
 block2_conv2_block2_conv2_kernel"
block2_conv2_block2_conv2_bias$
 block3_conv1_block3_conv1_kernel"
block3_conv1_block3_conv1_bias$
 block3_conv2_block3_conv2_kernel"
block3_conv2_block3_conv2_bias$
 block3_conv3_block3_conv3_kernel"
block3_conv3_block3_conv3_bias$
 block4_conv1_block4_conv1_kernel"
block4_conv1_block4_conv1_bias$
 block4_conv2_block4_conv2_kernel"
block4_conv2_block4_conv2_bias$
 block4_conv3_block4_conv3_kernel"
block4_conv3_block4_conv3_bias$
 block5_conv1_block5_conv1_kernel"
block5_conv1_block5_conv1_bias
sequential_kernel
sequential_bias
sequential_kernel_1
sequential_bias_1
sequential_kernel_2
sequential_bias_2
identity??$block1_conv1/StatefulPartitionedCall?$block1_conv2/StatefulPartitionedCall?$block2_conv1/StatefulPartitionedCall?$block2_conv2/StatefulPartitionedCall?$block3_conv1/StatefulPartitionedCall?$block3_conv2/StatefulPartitionedCall?$block3_conv3/StatefulPartitionedCall?$block4_conv1/StatefulPartitionedCall?$block4_conv2/StatefulPartitionedCall?$block4_conv3/StatefulPartitionedCall?$block5_conv1/StatefulPartitionedCall?"sequential/StatefulPartitionedCall?
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCallinputs block1_conv1_block1_conv1_kernelblock1_conv1_block1_conv1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@xx*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block1_conv1_layer_call_and_return_conditional_losses_16052&
$block1_conv1/StatefulPartitionedCall?
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0 block1_conv2_block1_conv2_kernelblock1_conv2_block1_conv2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@xx*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block1_conv2_layer_call_and_return_conditional_losses_16282&
$block1_conv2/StatefulPartitionedCall?
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@<<* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_block1_pool_layer_call_and_return_conditional_losses_13242
block1_pool/PartitionedCall?
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall$block1_pool/PartitionedCall:output:0 block2_conv1_block2_conv1_kernelblock2_conv1_block2_conv1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????<<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block2_conv1_layer_call_and_return_conditional_losses_16572&
$block2_conv1/StatefulPartitionedCall?
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0 block2_conv2_block2_conv2_kernelblock2_conv2_block2_conv2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????<<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block2_conv2_layer_call_and_return_conditional_losses_16802&
$block2_conv2/StatefulPartitionedCall?
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_block2_pool_layer_call_and_return_conditional_losses_13412
block2_pool/PartitionedCall?
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall$block2_pool/PartitionedCall:output:0 block3_conv1_block3_conv1_kernelblock3_conv1_block3_conv1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block3_conv1_layer_call_and_return_conditional_losses_17092&
$block3_conv1/StatefulPartitionedCall?
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0 block3_conv2_block3_conv2_kernelblock3_conv2_block3_conv2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block3_conv2_layer_call_and_return_conditional_losses_17322&
$block3_conv2/StatefulPartitionedCall?
$block3_conv3/StatefulPartitionedCallStatefulPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0 block3_conv3_block3_conv3_kernelblock3_conv3_block3_conv3_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block3_conv3_layer_call_and_return_conditional_losses_17552&
$block3_conv3/StatefulPartitionedCall?
block3_pool/PartitionedCallPartitionedCall-block3_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_block3_pool_layer_call_and_return_conditional_losses_13582
block3_pool/PartitionedCall?
$block4_conv1/StatefulPartitionedCallStatefulPartitionedCall$block3_pool/PartitionedCall:output:0 block4_conv1_block4_conv1_kernelblock4_conv1_block4_conv1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block4_conv1_layer_call_and_return_conditional_losses_17842&
$block4_conv1/StatefulPartitionedCall?
$block4_conv2/StatefulPartitionedCallStatefulPartitionedCall-block4_conv1/StatefulPartitionedCall:output:0 block4_conv2_block4_conv2_kernelblock4_conv2_block4_conv2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block4_conv2_layer_call_and_return_conditional_losses_18072&
$block4_conv2/StatefulPartitionedCall?
$block4_conv3/StatefulPartitionedCallStatefulPartitionedCall-block4_conv2/StatefulPartitionedCall:output:0 block4_conv3_block4_conv3_kernelblock4_conv3_block4_conv3_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block4_conv3_layer_call_and_return_conditional_losses_18302&
$block4_conv3/StatefulPartitionedCall?
block4_pool/PartitionedCallPartitionedCall-block4_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_block4_pool_layer_call_and_return_conditional_losses_13752
block4_pool/PartitionedCall?
$block5_conv1/StatefulPartitionedCallStatefulPartitionedCall$block4_pool/PartitionedCall:output:0 block5_conv1_block5_conv1_kernelblock5_conv1_block5_conv1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block5_conv1_layer_call_and_return_conditional_losses_18592&
$block5_conv1/StatefulPartitionedCall?
"sequential/StatefulPartitionedCallStatefulPartitionedCall-block5_conv1/StatefulPartitionedCall:output:0sequential_kernelsequential_biassequential_kernel_1sequential_bias_1sequential_kernel_2sequential_bias_2*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_19052$
"sequential/StatefulPartitionedCall?
IdentityIdentity+sequential/StatefulPartitionedCall:output:0%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall#^sequential/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????xx::::::::::::::::::::::::::::2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall2L
$block3_conv2/StatefulPartitionedCall$block3_conv2/StatefulPartitionedCall2L
$block3_conv3/StatefulPartitionedCall$block3_conv3/StatefulPartitionedCall2L
$block4_conv1/StatefulPartitionedCall$block4_conv1/StatefulPartitionedCall2L
$block4_conv2/StatefulPartitionedCall$block4_conv2/StatefulPartitionedCall2L
$block4_conv3/StatefulPartitionedCall$block4_conv3/StatefulPartitionedCall2L
$block5_conv1/StatefulPartitionedCall$block5_conv1/StatefulPartitionedCall2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall:W S
/
_output_shapes
:?????????xx
 
_user_specified_nameinputs
?
?
+__inference_block2_conv2_layer_call_fn_2579

inputs
block2_conv2_kernel
block2_conv2_bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsblock2_conv2_kernelblock2_conv2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????<<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_block2_conv2_layer_call_and_return_conditional_losses_16802
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????<<2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????<<::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????<<
 
_user_specified_nameinputs
?
a
E__inference_block1_pool_layer_call_and_return_conditional_losses_1316

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
data_formatNCHW*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
F__inference_block4_conv3_layer_call_and_return_conditional_losses_1830

inputs-
)conv2d_readvariableop_block4_conv3_kernel,
(biasadd_readvariableop_block4_conv3_bias
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOp)conv2d_readvariableop_block4_conv3_kernel*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2D	MLCConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
data_formatNCHW*
num_args *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOp(biasadd_readvariableop_block4_conv3_bias*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
data_formatNCHW2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
__inference__wrapped_model_1310
input_1B
>model_1_block1_conv1_conv2d_readvariableop_block1_conv1_kernelA
=model_1_block1_conv1_biasadd_readvariableop_block1_conv1_biasB
>model_1_block1_conv2_conv2d_readvariableop_block1_conv2_kernelA
=model_1_block1_conv2_biasadd_readvariableop_block1_conv2_biasB
>model_1_block2_conv1_conv2d_readvariableop_block2_conv1_kernelA
=model_1_block2_conv1_biasadd_readvariableop_block2_conv1_biasB
>model_1_block2_conv2_conv2d_readvariableop_block2_conv2_kernelA
=model_1_block2_conv2_biasadd_readvariableop_block2_conv2_biasB
>model_1_block3_conv1_conv2d_readvariableop_block3_conv1_kernelA
=model_1_block3_conv1_biasadd_readvariableop_block3_conv1_biasB
>model_1_block3_conv2_conv2d_readvariableop_block3_conv2_kernelA
=model_1_block3_conv2_biasadd_readvariableop_block3_conv2_biasB
>model_1_block3_conv3_conv2d_readvariableop_block3_conv3_kernelA
=model_1_block3_conv3_biasadd_readvariableop_block3_conv3_biasB
>model_1_block4_conv1_conv2d_readvariableop_block4_conv1_kernelA
=model_1_block4_conv1_biasadd_readvariableop_block4_conv1_biasB
>model_1_block4_conv2_conv2d_readvariableop_block4_conv2_kernelA
=model_1_block4_conv2_biasadd_readvariableop_block4_conv2_biasB
>model_1_block4_conv3_conv2d_readvariableop_block4_conv3_kernelA
=model_1_block4_conv3_biasadd_readvariableop_block4_conv3_biasB
>model_1_block5_conv1_conv2d_readvariableop_block5_conv1_kernelA
=model_1_block5_conv1_biasadd_readvariableop_block5_conv1_bias:
6model_1_sequential_conv2d_conv2d_readvariableop_kernel9
5model_1_sequential_conv2d_biasadd_readvariableop_bias>
:model_1_sequential_dense_mlcmatmul_readvariableop_kernel_1:
6model_1_sequential_dense_biasadd_readvariableop_bias_1@
<model_1_sequential_dense_1_mlcmatmul_readvariableop_kernel_2<
8model_1_sequential_dense_1_biasadd_readvariableop_bias_2
identity??+model_1/block1_conv1/BiasAdd/ReadVariableOp?*model_1/block1_conv1/Conv2D/ReadVariableOp?+model_1/block1_conv2/BiasAdd/ReadVariableOp?*model_1/block1_conv2/Conv2D/ReadVariableOp?+model_1/block2_conv1/BiasAdd/ReadVariableOp?*model_1/block2_conv1/Conv2D/ReadVariableOp?+model_1/block2_conv2/BiasAdd/ReadVariableOp?*model_1/block2_conv2/Conv2D/ReadVariableOp?+model_1/block3_conv1/BiasAdd/ReadVariableOp?*model_1/block3_conv1/Conv2D/ReadVariableOp?+model_1/block3_conv2/BiasAdd/ReadVariableOp?*model_1/block3_conv2/Conv2D/ReadVariableOp?+model_1/block3_conv3/BiasAdd/ReadVariableOp?*model_1/block3_conv3/Conv2D/ReadVariableOp?+model_1/block4_conv1/BiasAdd/ReadVariableOp?*model_1/block4_conv1/Conv2D/ReadVariableOp?+model_1/block4_conv2/BiasAdd/ReadVariableOp?*model_1/block4_conv2/Conv2D/ReadVariableOp?+model_1/block4_conv3/BiasAdd/ReadVariableOp?*model_1/block4_conv3/Conv2D/ReadVariableOp?+model_1/block5_conv1/BiasAdd/ReadVariableOp?*model_1/block5_conv1/Conv2D/ReadVariableOp?0model_1/sequential/conv2d/BiasAdd/ReadVariableOp?/model_1/sequential/conv2d/Conv2D/ReadVariableOp?/model_1/sequential/dense/BiasAdd/ReadVariableOp?1model_1/sequential/dense/MLCMatMul/ReadVariableOp?1model_1/sequential/dense_1/BiasAdd/ReadVariableOp?3model_1/sequential/dense_1/MLCMatMul/ReadVariableOp?
*model_1/block1_conv1/Conv2D/ReadVariableOpReadVariableOp>model_1_block1_conv1_conv2d_readvariableop_block1_conv1_kernel*&
_output_shapes
:@*
dtype02,
*model_1/block1_conv1/Conv2D/ReadVariableOp?
model_1/block1_conv1/Conv2D	MLCConv2Dinput_12model_1/block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@xx*
data_formatNCHW*
num_args *
paddingSAME*
strides
2
model_1/block1_conv1/Conv2D?
+model_1/block1_conv1/BiasAdd/ReadVariableOpReadVariableOp=model_1_block1_conv1_biasadd_readvariableop_block1_conv1_bias*
_output_shapes
:@*
dtype02-
+model_1/block1_conv1/BiasAdd/ReadVariableOp?
model_1/block1_conv1/BiasAddBiasAdd$model_1/block1_conv1/Conv2D:output:03model_1/block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@xx*
data_formatNCHW2
model_1/block1_conv1/BiasAdd?
model_1/block1_conv1/ReluRelu%model_1/block1_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@xx2
model_1/block1_conv1/Relu?
*model_1/block1_conv2/Conv2D/ReadVariableOpReadVariableOp>model_1_block1_conv2_conv2d_readvariableop_block1_conv2_kernel*&
_output_shapes
:@@*
dtype02,
*model_1/block1_conv2/Conv2D/ReadVariableOp?
model_1/block1_conv2/Conv2D	MLCConv2D'model_1/block1_conv1/Relu:activations:02model_1/block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@xx*
data_formatNCHW*
num_args *
paddingSAME*
strides
2
model_1/block1_conv2/Conv2D?
+model_1/block1_conv2/BiasAdd/ReadVariableOpReadVariableOp=model_1_block1_conv2_biasadd_readvariableop_block1_conv2_bias*
_output_shapes
:@*
dtype02-
+model_1/block1_conv2/BiasAdd/ReadVariableOp?
model_1/block1_conv2/BiasAddBiasAdd$model_1/block1_conv2/Conv2D:output:03model_1/block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@xx*
data_formatNCHW2
model_1/block1_conv2/BiasAdd?
model_1/block1_conv2/ReluRelu%model_1/block1_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@xx2
model_1/block1_conv2/Relu?
model_1/block1_pool/MaxPoolMaxPool'model_1/block1_conv2/Relu:activations:0*/
_output_shapes
:?????????@<<*
data_formatNCHW*
ksize
*
paddingVALID*
strides
2
model_1/block1_pool/MaxPool?
*model_1/block2_conv1/Conv2D/ReadVariableOpReadVariableOp>model_1_block2_conv1_conv2d_readvariableop_block2_conv1_kernel*'
_output_shapes
:@?*
dtype02,
*model_1/block2_conv1/Conv2D/ReadVariableOp?
model_1/block2_conv1/Conv2D	MLCConv2D$model_1/block1_pool/MaxPool:output:02model_1/block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????<<*
data_formatNCHW*
num_args *
paddingSAME*
strides
2
model_1/block2_conv1/Conv2D?
+model_1/block2_conv1/BiasAdd/ReadVariableOpReadVariableOp=model_1_block2_conv1_biasadd_readvariableop_block2_conv1_bias*
_output_shapes	
:?*
dtype02-
+model_1/block2_conv1/BiasAdd/ReadVariableOp?
model_1/block2_conv1/BiasAddBiasAdd$model_1/block2_conv1/Conv2D:output:03model_1/block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????<<*
data_formatNCHW2
model_1/block2_conv1/BiasAdd?
model_1/block2_conv1/ReluRelu%model_1/block2_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:??????????<<2
model_1/block2_conv1/Relu?
*model_1/block2_conv2/Conv2D/ReadVariableOpReadVariableOp>model_1_block2_conv2_conv2d_readvariableop_block2_conv2_kernel*(
_output_shapes
:??*
dtype02,
*model_1/block2_conv2/Conv2D/ReadVariableOp?
model_1/block2_conv2/Conv2D	MLCConv2D'model_1/block2_conv1/Relu:activations:02model_1/block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????<<*
data_formatNCHW*
num_args *
paddingSAME*
strides
2
model_1/block2_conv2/Conv2D?
+model_1/block2_conv2/BiasAdd/ReadVariableOpReadVariableOp=model_1_block2_conv2_biasadd_readvariableop_block2_conv2_bias*
_output_shapes	
:?*
dtype02-
+model_1/block2_conv2/BiasAdd/ReadVariableOp?
model_1/block2_conv2/BiasAddBiasAdd$model_1/block2_conv2/Conv2D:output:03model_1/block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????<<*
data_formatNCHW2
model_1/block2_conv2/BiasAdd?
model_1/block2_conv2/ReluRelu%model_1/block2_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:??????????<<2
model_1/block2_conv2/Relu?
model_1/block2_pool/MaxPoolMaxPool'model_1/block2_conv2/Relu:activations:0*0
_output_shapes
:??????????*
data_formatNCHW*
ksize
*
paddingVALID*
strides
2
model_1/block2_pool/MaxPool?
*model_1/block3_conv1/Conv2D/ReadVariableOpReadVariableOp>model_1_block3_conv1_conv2d_readvariableop_block3_conv1_kernel*(
_output_shapes
:??*
dtype02,
*model_1/block3_conv1/Conv2D/ReadVariableOp?
model_1/block3_conv1/Conv2D	MLCConv2D$model_1/block2_pool/MaxPool:output:02model_1/block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
data_formatNCHW*
num_args *
paddingSAME*
strides
2
model_1/block3_conv1/Conv2D?
+model_1/block3_conv1/BiasAdd/ReadVariableOpReadVariableOp=model_1_block3_conv1_biasadd_readvariableop_block3_conv1_bias*
_output_shapes	
:?*
dtype02-
+model_1/block3_conv1/BiasAdd/ReadVariableOp?
model_1/block3_conv1/BiasAddBiasAdd$model_1/block3_conv1/Conv2D:output:03model_1/block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
data_formatNCHW2
model_1/block3_conv1/BiasAdd?
model_1/block3_conv1/ReluRelu%model_1/block3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
model_1/block3_conv1/Relu?
*model_1/block3_conv2/Conv2D/ReadVariableOpReadVariableOp>model_1_block3_conv2_conv2d_readvariableop_block3_conv2_kernel*(
_output_shapes
:??*
dtype02,
*model_1/block3_conv2/Conv2D/ReadVariableOp?
model_1/block3_conv2/Conv2D	MLCConv2D'model_1/block3_conv1/Relu:activations:02model_1/block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
data_formatNCHW*
num_args *
paddingSAME*
strides
2
model_1/block3_conv2/Conv2D?
+model_1/block3_conv2/BiasAdd/ReadVariableOpReadVariableOp=model_1_block3_conv2_biasadd_readvariableop_block3_conv2_bias*
_output_shapes	
:?*
dtype02-
+model_1/block3_conv2/BiasAdd/ReadVariableOp?
model_1/block3_conv2/BiasAddBiasAdd$model_1/block3_conv2/Conv2D:output:03model_1/block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
data_formatNCHW2
model_1/block3_conv2/BiasAdd?
model_1/block3_conv2/ReluRelu%model_1/block3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
model_1/block3_conv2/Relu?
*model_1/block3_conv3/Conv2D/ReadVariableOpReadVariableOp>model_1_block3_conv3_conv2d_readvariableop_block3_conv3_kernel*(
_output_shapes
:??*
dtype02,
*model_1/block3_conv3/Conv2D/ReadVariableOp?
model_1/block3_conv3/Conv2D	MLCConv2D'model_1/block3_conv2/Relu:activations:02model_1/block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
data_formatNCHW*
num_args *
paddingSAME*
strides
2
model_1/block3_conv3/Conv2D?
+model_1/block3_conv3/BiasAdd/ReadVariableOpReadVariableOp=model_1_block3_conv3_biasadd_readvariableop_block3_conv3_bias*
_output_shapes	
:?*
dtype02-
+model_1/block3_conv3/BiasAdd/ReadVariableOp?
model_1/block3_conv3/BiasAddBiasAdd$model_1/block3_conv3/Conv2D:output:03model_1/block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
data_formatNCHW2
model_1/block3_conv3/BiasAdd?
model_1/block3_conv3/ReluRelu%model_1/block3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
model_1/block3_conv3/Relu?
model_1/block3_pool/MaxPoolMaxPool'model_1/block3_conv3/Relu:activations:0*0
_output_shapes
:??????????*
data_formatNCHW*
ksize
*
paddingVALID*
strides
2
model_1/block3_pool/MaxPool?
*model_1/block4_conv1/Conv2D/ReadVariableOpReadVariableOp>model_1_block4_conv1_conv2d_readvariableop_block4_conv1_kernel*(
_output_shapes
:??*
dtype02,
*model_1/block4_conv1/Conv2D/ReadVariableOp?
model_1/block4_conv1/Conv2D	MLCConv2D$model_1/block3_pool/MaxPool:output:02model_1/block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
data_formatNCHW*
num_args *
paddingSAME*
strides
2
model_1/block4_conv1/Conv2D?
+model_1/block4_conv1/BiasAdd/ReadVariableOpReadVariableOp=model_1_block4_conv1_biasadd_readvariableop_block4_conv1_bias*
_output_shapes	
:?*
dtype02-
+model_1/block4_conv1/BiasAdd/ReadVariableOp?
model_1/block4_conv1/BiasAddBiasAdd$model_1/block4_conv1/Conv2D:output:03model_1/block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
data_formatNCHW2
model_1/block4_conv1/BiasAdd?
model_1/block4_conv1/ReluRelu%model_1/block4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
model_1/block4_conv1/Relu?
*model_1/block4_conv2/Conv2D/ReadVariableOpReadVariableOp>model_1_block4_conv2_conv2d_readvariableop_block4_conv2_kernel*(
_output_shapes
:??*
dtype02,
*model_1/block4_conv2/Conv2D/ReadVariableOp?
model_1/block4_conv2/Conv2D	MLCConv2D'model_1/block4_conv1/Relu:activations:02model_1/block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
data_formatNCHW*
num_args *
paddingSAME*
strides
2
model_1/block4_conv2/Conv2D?
+model_1/block4_conv2/BiasAdd/ReadVariableOpReadVariableOp=model_1_block4_conv2_biasadd_readvariableop_block4_conv2_bias*
_output_shapes	
:?*
dtype02-
+model_1/block4_conv2/BiasAdd/ReadVariableOp?
model_1/block4_conv2/BiasAddBiasAdd$model_1/block4_conv2/Conv2D:output:03model_1/block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
data_formatNCHW2
model_1/block4_conv2/BiasAdd?
model_1/block4_conv2/ReluRelu%model_1/block4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
model_1/block4_conv2/Relu?
*model_1/block4_conv3/Conv2D/ReadVariableOpReadVariableOp>model_1_block4_conv3_conv2d_readvariableop_block4_conv3_kernel*(
_output_shapes
:??*
dtype02,
*model_1/block4_conv3/Conv2D/ReadVariableOp?
model_1/block4_conv3/Conv2D	MLCConv2D'model_1/block4_conv2/Relu:activations:02model_1/block4_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
data_formatNCHW*
num_args *
paddingSAME*
strides
2
model_1/block4_conv3/Conv2D?
+model_1/block4_conv3/BiasAdd/ReadVariableOpReadVariableOp=model_1_block4_conv3_biasadd_readvariableop_block4_conv3_bias*
_output_shapes	
:?*
dtype02-
+model_1/block4_conv3/BiasAdd/ReadVariableOp?
model_1/block4_conv3/BiasAddBiasAdd$model_1/block4_conv3/Conv2D:output:03model_1/block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
data_formatNCHW2
model_1/block4_conv3/BiasAdd?
model_1/block4_conv3/ReluRelu%model_1/block4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
model_1/block4_conv3/Relu?
model_1/block4_pool/MaxPoolMaxPool'model_1/block4_conv3/Relu:activations:0*0
_output_shapes
:??????????*
data_formatNCHW*
ksize
*
paddingVALID*
strides
2
model_1/block4_pool/MaxPool?
*model_1/block5_conv1/Conv2D/ReadVariableOpReadVariableOp>model_1_block5_conv1_conv2d_readvariableop_block5_conv1_kernel*(
_output_shapes
:??*
dtype02,
*model_1/block5_conv1/Conv2D/ReadVariableOp?
model_1/block5_conv1/Conv2D	MLCConv2D$model_1/block4_pool/MaxPool:output:02model_1/block5_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
data_formatNCHW*
num_args *
paddingSAME*
strides
2
model_1/block5_conv1/Conv2D?
+model_1/block5_conv1/BiasAdd/ReadVariableOpReadVariableOp=model_1_block5_conv1_biasadd_readvariableop_block5_conv1_bias*
_output_shapes	
:?*
dtype02-
+model_1/block5_conv1/BiasAdd/ReadVariableOp?
model_1/block5_conv1/BiasAddBiasAdd$model_1/block5_conv1/Conv2D:output:03model_1/block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
data_formatNCHW2
model_1/block5_conv1/BiasAdd?
model_1/block5_conv1/ReluRelu%model_1/block5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
model_1/block5_conv1/Relu?
/model_1/sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp6model_1_sequential_conv2d_conv2d_readvariableop_kernel*'
_output_shapes
:?*
dtype021
/model_1/sequential/conv2d/Conv2D/ReadVariableOp?
 model_1/sequential/conv2d/Conv2D	MLCConv2D'model_1/block5_conv1/Relu:activations:07model_1/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
data_formatNCHW*
num_args *
paddingVALID*
strides
2"
 model_1/sequential/conv2d/Conv2D?
0model_1/sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp5model_1_sequential_conv2d_biasadd_readvariableop_bias*
_output_shapes
:*
dtype022
0model_1/sequential/conv2d/BiasAdd/ReadVariableOp?
!model_1/sequential/conv2d/BiasAddBiasAdd)model_1/sequential/conv2d/Conv2D:output:08model_1/sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
data_formatNCHW2#
!model_1/sequential/conv2d/BiasAdd?
model_1/sequential/conv2d/ReluRelu*model_1/sequential/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2 
model_1/sequential/conv2d/Relu?
(model_1/sequential/max_pooling2d/MaxPoolMaxPool,model_1/sequential/conv2d/Relu:activations:0*/
_output_shapes
:?????????*
data_formatNCHW*
ksize
*
paddingVALID*
strides
2*
(model_1/sequential/max_pooling2d/MaxPool?
)model_1/sequential/flatten/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2+
)model_1/sequential/flatten/transpose/perm?
$model_1/sequential/flatten/transpose	Transpose1model_1/sequential/max_pooling2d/MaxPool:output:02model_1/sequential/flatten/transpose/perm:output:0*
T0*/
_output_shapes
:?????????2&
$model_1/sequential/flatten/transpose?
 model_1/sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????   2"
 model_1/sequential/flatten/Const?
"model_1/sequential/flatten/ReshapeReshape(model_1/sequential/flatten/transpose:y:0)model_1/sequential/flatten/Const:output:0*
T0*(
_output_shapes
:??????????2$
"model_1/sequential/flatten/Reshape?
#model_1/sequential/dropout/IdentityIdentity+model_1/sequential/flatten/Reshape:output:0*
T0*(
_output_shapes
:??????????2%
#model_1/sequential/dropout/Identity?
1model_1/sequential/dense/MLCMatMul/ReadVariableOpReadVariableOp:model_1_sequential_dense_mlcmatmul_readvariableop_kernel_1*
_output_shapes
:	?2*
dtype023
1model_1/sequential/dense/MLCMatMul/ReadVariableOp?
"model_1/sequential/dense/MLCMatMul	MLCMatMul,model_1/sequential/dropout/Identity:output:09model_1/sequential/dense/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22$
"model_1/sequential/dense/MLCMatMul?
/model_1/sequential/dense/BiasAdd/ReadVariableOpReadVariableOp6model_1_sequential_dense_biasadd_readvariableop_bias_1*
_output_shapes
:2*
dtype021
/model_1/sequential/dense/BiasAdd/ReadVariableOp?
 model_1/sequential/dense/BiasAddBiasAdd,model_1/sequential/dense/MLCMatMul:product:07model_1/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????22"
 model_1/sequential/dense/BiasAdd?
model_1/sequential/dense/ReluRelu)model_1/sequential/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????22
model_1/sequential/dense/Relu?
3model_1/sequential/dense_1/MLCMatMul/ReadVariableOpReadVariableOp<model_1_sequential_dense_1_mlcmatmul_readvariableop_kernel_2*
_output_shapes

:2*
dtype025
3model_1/sequential/dense_1/MLCMatMul/ReadVariableOp?
$model_1/sequential/dense_1/MLCMatMul	MLCMatMul+model_1/sequential/dense/Relu:activations:0;model_1/sequential/dense_1/MLCMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2&
$model_1/sequential/dense_1/MLCMatMul?
1model_1/sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp8model_1_sequential_dense_1_biasadd_readvariableop_bias_2*
_output_shapes
:*
dtype023
1model_1/sequential/dense_1/BiasAdd/ReadVariableOp?
"model_1/sequential/dense_1/BiasAddBiasAdd.model_1/sequential/dense_1/MLCMatMul:product:09model_1/sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2$
"model_1/sequential/dense_1/BiasAdd?
"model_1/sequential/dense_1/SigmoidSigmoid+model_1/sequential/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2$
"model_1/sequential/dense_1/Sigmoid?
IdentityIdentity&model_1/sequential/dense_1/Sigmoid:y:0,^model_1/block1_conv1/BiasAdd/ReadVariableOp+^model_1/block1_conv1/Conv2D/ReadVariableOp,^model_1/block1_conv2/BiasAdd/ReadVariableOp+^model_1/block1_conv2/Conv2D/ReadVariableOp,^model_1/block2_conv1/BiasAdd/ReadVariableOp+^model_1/block2_conv1/Conv2D/ReadVariableOp,^model_1/block2_conv2/BiasAdd/ReadVariableOp+^model_1/block2_conv2/Conv2D/ReadVariableOp,^model_1/block3_conv1/BiasAdd/ReadVariableOp+^model_1/block3_conv1/Conv2D/ReadVariableOp,^model_1/block3_conv2/BiasAdd/ReadVariableOp+^model_1/block3_conv2/Conv2D/ReadVariableOp,^model_1/block3_conv3/BiasAdd/ReadVariableOp+^model_1/block3_conv3/Conv2D/ReadVariableOp,^model_1/block4_conv1/BiasAdd/ReadVariableOp+^model_1/block4_conv1/Conv2D/ReadVariableOp,^model_1/block4_conv2/BiasAdd/ReadVariableOp+^model_1/block4_conv2/Conv2D/ReadVariableOp,^model_1/block4_conv3/BiasAdd/ReadVariableOp+^model_1/block4_conv3/Conv2D/ReadVariableOp,^model_1/block5_conv1/BiasAdd/ReadVariableOp+^model_1/block5_conv1/Conv2D/ReadVariableOp1^model_1/sequential/conv2d/BiasAdd/ReadVariableOp0^model_1/sequential/conv2d/Conv2D/ReadVariableOp0^model_1/sequential/dense/BiasAdd/ReadVariableOp2^model_1/sequential/dense/MLCMatMul/ReadVariableOp2^model_1/sequential/dense_1/BiasAdd/ReadVariableOp4^model_1/sequential/dense_1/MLCMatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????xx::::::::::::::::::::::::::::2Z
+model_1/block1_conv1/BiasAdd/ReadVariableOp+model_1/block1_conv1/BiasAdd/ReadVariableOp2X
*model_1/block1_conv1/Conv2D/ReadVariableOp*model_1/block1_conv1/Conv2D/ReadVariableOp2Z
+model_1/block1_conv2/BiasAdd/ReadVariableOp+model_1/block1_conv2/BiasAdd/ReadVariableOp2X
*model_1/block1_conv2/Conv2D/ReadVariableOp*model_1/block1_conv2/Conv2D/ReadVariableOp2Z
+model_1/block2_conv1/BiasAdd/ReadVariableOp+model_1/block2_conv1/BiasAdd/ReadVariableOp2X
*model_1/block2_conv1/Conv2D/ReadVariableOp*model_1/block2_conv1/Conv2D/ReadVariableOp2Z
+model_1/block2_conv2/BiasAdd/ReadVariableOp+model_1/block2_conv2/BiasAdd/ReadVariableOp2X
*model_1/block2_conv2/Conv2D/ReadVariableOp*model_1/block2_conv2/Conv2D/ReadVariableOp2Z
+model_1/block3_conv1/BiasAdd/ReadVariableOp+model_1/block3_conv1/BiasAdd/ReadVariableOp2X
*model_1/block3_conv1/Conv2D/ReadVariableOp*model_1/block3_conv1/Conv2D/ReadVariableOp2Z
+model_1/block3_conv2/BiasAdd/ReadVariableOp+model_1/block3_conv2/BiasAdd/ReadVariableOp2X
*model_1/block3_conv2/Conv2D/ReadVariableOp*model_1/block3_conv2/Conv2D/ReadVariableOp2Z
+model_1/block3_conv3/BiasAdd/ReadVariableOp+model_1/block3_conv3/BiasAdd/ReadVariableOp2X
*model_1/block3_conv3/Conv2D/ReadVariableOp*model_1/block3_conv3/Conv2D/ReadVariableOp2Z
+model_1/block4_conv1/BiasAdd/ReadVariableOp+model_1/block4_conv1/BiasAdd/ReadVariableOp2X
*model_1/block4_conv1/Conv2D/ReadVariableOp*model_1/block4_conv1/Conv2D/ReadVariableOp2Z
+model_1/block4_conv2/BiasAdd/ReadVariableOp+model_1/block4_conv2/BiasAdd/ReadVariableOp2X
*model_1/block4_conv2/Conv2D/ReadVariableOp*model_1/block4_conv2/Conv2D/ReadVariableOp2Z
+model_1/block4_conv3/BiasAdd/ReadVariableOp+model_1/block4_conv3/BiasAdd/ReadVariableOp2X
*model_1/block4_conv3/Conv2D/ReadVariableOp*model_1/block4_conv3/Conv2D/ReadVariableOp2Z
+model_1/block5_conv1/BiasAdd/ReadVariableOp+model_1/block5_conv1/BiasAdd/ReadVariableOp2X
*model_1/block5_conv1/Conv2D/ReadVariableOp*model_1/block5_conv1/Conv2D/ReadVariableOp2d
0model_1/sequential/conv2d/BiasAdd/ReadVariableOp0model_1/sequential/conv2d/BiasAdd/ReadVariableOp2b
/model_1/sequential/conv2d/Conv2D/ReadVariableOp/model_1/sequential/conv2d/Conv2D/ReadVariableOp2b
/model_1/sequential/dense/BiasAdd/ReadVariableOp/model_1/sequential/dense/BiasAdd/ReadVariableOp2f
1model_1/sequential/dense/MLCMatMul/ReadVariableOp1model_1/sequential/dense/MLCMatMul/ReadVariableOp2f
1model_1/sequential/dense_1/BiasAdd/ReadVariableOp1model_1/sequential/dense_1/BiasAdd/ReadVariableOp2j
3model_1/sequential/dense_1/MLCMatMul/ReadVariableOp3model_1/sequential/dense_1/MLCMatMul/ReadVariableOp:X T
/
_output_shapes
:?????????xx
!
_user_specified_name	input_1"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
C
input_18
serving_default_input_1:0?????????xx>

sequential0
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
??
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer_with_weights-6

layer-9
layer-10
layer_with_weights-7
layer-11
layer_with_weights-8
layer-12
layer_with_weights-9
layer-13
layer-14
layer_with_weights-10
layer-15
layer_with_weights-11
layer-16
trainable_variables
	variables
regularization_losses
	keras_api

signatures
?_default_save_signature
?__call__
+?&call_and_return_all_conditional_losses"??
_tf_keras_network??{"class_name": "Functional", "name": "model_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 120, 120]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "block1_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block1_conv1", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block1_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block1_conv2", "inbound_nodes": [[["block1_conv1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block1_pool", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_first"}, "name": "block1_pool", "inbound_nodes": [[["block1_conv2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block2_conv1", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block2_conv1", "inbound_nodes": [[["block1_pool", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block2_conv2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block2_conv2", "inbound_nodes": [[["block2_conv1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block2_pool", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_first"}, "name": "block2_pool", "inbound_nodes": [[["block2_conv2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block3_conv1", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block3_conv1", "inbound_nodes": [[["block2_pool", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block3_conv2", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block3_conv2", "inbound_nodes": [[["block3_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block3_conv3", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block3_conv3", "inbound_nodes": [[["block3_conv2", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block3_pool", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_first"}, "name": "block3_pool", "inbound_nodes": [[["block3_conv3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block4_conv1", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block4_conv1", "inbound_nodes": [[["block3_pool", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block4_conv2", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block4_conv2", "inbound_nodes": [[["block4_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block4_conv3", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block4_conv3", "inbound_nodes": [[["block4_conv2", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block4_pool", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_first"}, "name": "block4_pool", "inbound_nodes": [[["block4_conv3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block5_conv1", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block5_conv1", "inbound_nodes": [[["block4_pool", 0, 0, {}]]]}, {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 512, 7, 7]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_first"}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_first"}}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "name": "sequential", "inbound_nodes": [[["block5_conv1", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["sequential", 1, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 3, 120, 120]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 120, 120]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 120, 120]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "block1_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block1_conv1", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block1_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block1_conv2", "inbound_nodes": [[["block1_conv1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block1_pool", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_first"}, "name": "block1_pool", "inbound_nodes": [[["block1_conv2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block2_conv1", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block2_conv1", "inbound_nodes": [[["block1_pool", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block2_conv2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block2_conv2", "inbound_nodes": [[["block2_conv1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block2_pool", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_first"}, "name": "block2_pool", "inbound_nodes": [[["block2_conv2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block3_conv1", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block3_conv1", "inbound_nodes": [[["block2_pool", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block3_conv2", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block3_conv2", "inbound_nodes": [[["block3_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block3_conv3", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block3_conv3", "inbound_nodes": [[["block3_conv2", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block3_pool", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_first"}, "name": "block3_pool", "inbound_nodes": [[["block3_conv3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block4_conv1", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block4_conv1", "inbound_nodes": [[["block3_pool", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block4_conv2", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block4_conv2", "inbound_nodes": [[["block4_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block4_conv3", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block4_conv3", "inbound_nodes": [[["block4_conv2", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block4_pool", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_first"}, "name": "block4_pool", "inbound_nodes": [[["block4_conv3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block5_conv1", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block5_conv1", "inbound_nodes": [[["block4_pool", 0, 0, {}]]]}, {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 512, 7, 7]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_first"}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_first"}}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "name": "sequential", "inbound_nodes": [[["block5_conv1", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["sequential", 1, 0]]}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 120, 120]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 120, 120]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
?	

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "block1_conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "block1_conv1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 120, 120]}}
?	

kernel
bias
trainable_variables
 	variables
!regularization_losses
"	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "block1_conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "block1_conv2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 120, 120]}}
?
#trainable_variables
$	variables
%regularization_losses
&	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "block1_pool", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "block1_pool", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_first"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?	

'kernel
(bias
)trainable_variables
*	variables
+regularization_losses
,	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "block2_conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "block2_conv1", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 60, 60]}}
?


-kernel
.bias
/trainable_variables
0	variables
1regularization_losses
2	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "block2_conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "block2_conv2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 60, 60]}}
?
3trainable_variables
4	variables
5regularization_losses
6	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "block2_pool", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "block2_pool", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_first"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?


7kernel
8bias
9trainable_variables
:	variables
;regularization_losses
<	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "block3_conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "block3_conv1", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 30, 30]}}
?


=kernel
>bias
?trainable_variables
@	variables
Aregularization_losses
B	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "block3_conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "block3_conv2", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256, 30, 30]}}
?


Ckernel
Dbias
Etrainable_variables
F	variables
Gregularization_losses
H	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "block3_conv3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "block3_conv3", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256, 30, 30]}}
?
Itrainable_variables
J	variables
Kregularization_losses
L	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "block3_pool", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "block3_pool", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_first"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?


Mkernel
Nbias
Otrainable_variables
P	variables
Qregularization_losses
R	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "block4_conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "block4_conv1", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256, 15, 15]}}
?


Skernel
Tbias
Utrainable_variables
V	variables
Wregularization_losses
X	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "block4_conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "block4_conv2", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512, 15, 15]}}
?


Ykernel
Zbias
[trainable_variables
\	variables
]regularization_losses
^	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "block4_conv3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "block4_conv3", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512, 15, 15]}}
?
_trainable_variables
`	variables
aregularization_losses
b	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "block4_pool", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "block4_pool", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_first"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?	

ckernel
dbias
etrainable_variables
f	variables
gregularization_losses
h	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "block5_conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "block5_conv1", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512, 7, 7]}}
?.
ilayer_with_weights-0
ilayer-0
jlayer-1
klayer-2
llayer-3
mlayer_with_weights-1
mlayer-4
nlayer_with_weights-2
nlayer-5
o	optimizer
p
signatures
#q_self_saveable_object_factories
rtrainable_variables
s	variables
tregularization_losses
u	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?+
_tf_keras_sequential?+{"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 512, 7, 7]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_first"}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_first"}}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512, 7, 7]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 512, 7, 7]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_first"}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_first"}}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "binary_crossentropy", "metrics": ["accuracy"], "loss_weights": null, "sample_weight_mode": null, "weighted_metrics": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?
0
1
2
3
'4
(5
-6
.7
78
89
=10
>11
C12
D13
M14
N15
S16
T17
Y18
Z19
c20
d21
v22
w23
x24
y25
z26
{27"
trackable_list_wrapper
?
0
1
2
3
'4
(5
-6
.7
78
89
=10
>11
C12
D13
M14
N15
S16
T17
Y18
Z19
c20
d21
v22
w23
x24
y25
z26
{27"
trackable_list_wrapper
 "
trackable_list_wrapper
?

|layers
trainable_variables
}layer_regularization_losses
~metrics
layer_metrics
?non_trainable_variables
	variables
regularization_losses
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
-:+@2block1_conv1/kernel
:@2block1_conv1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
trainable_variables
 ?layer_regularization_losses
?metrics
?layer_metrics
?non_trainable_variables
	variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-:+@@2block1_conv2/kernel
:@2block1_conv2/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
trainable_variables
 ?layer_regularization_losses
?metrics
?layer_metrics
?non_trainable_variables
 	variables
!regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
#trainable_variables
 ?layer_regularization_losses
?metrics
?layer_metrics
?non_trainable_variables
$	variables
%regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.:,@?2block2_conv1/kernel
 :?2block2_conv1/bias
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
)trainable_variables
 ?layer_regularization_losses
?metrics
?layer_metrics
?non_trainable_variables
*	variables
+regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
/:-??2block2_conv2/kernel
 :?2block2_conv2/bias
.
-0
.1"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
/trainable_variables
 ?layer_regularization_losses
?metrics
?layer_metrics
?non_trainable_variables
0	variables
1regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
3trainable_variables
 ?layer_regularization_losses
?metrics
?layer_metrics
?non_trainable_variables
4	variables
5regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
/:-??2block3_conv1/kernel
 :?2block3_conv1/bias
.
70
81"
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
9trainable_variables
 ?layer_regularization_losses
?metrics
?layer_metrics
?non_trainable_variables
:	variables
;regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
/:-??2block3_conv2/kernel
 :?2block3_conv2/bias
.
=0
>1"
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?trainable_variables
 ?layer_regularization_losses
?metrics
?layer_metrics
?non_trainable_variables
@	variables
Aregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
/:-??2block3_conv3/kernel
 :?2block3_conv3/bias
.
C0
D1"
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
Etrainable_variables
 ?layer_regularization_losses
?metrics
?layer_metrics
?non_trainable_variables
F	variables
Gregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
Itrainable_variables
 ?layer_regularization_losses
?metrics
?layer_metrics
?non_trainable_variables
J	variables
Kregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
/:-??2block4_conv1/kernel
 :?2block4_conv1/bias
.
M0
N1"
trackable_list_wrapper
.
M0
N1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
Otrainable_variables
 ?layer_regularization_losses
?metrics
?layer_metrics
?non_trainable_variables
P	variables
Qregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
/:-??2block4_conv2/kernel
 :?2block4_conv2/bias
.
S0
T1"
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
Utrainable_variables
 ?layer_regularization_losses
?metrics
?layer_metrics
?non_trainable_variables
V	variables
Wregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
/:-??2block4_conv3/kernel
 :?2block4_conv3/bias
.
Y0
Z1"
trackable_list_wrapper
.
Y0
Z1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
[trainable_variables
 ?layer_regularization_losses
?metrics
?layer_metrics
?non_trainable_variables
\	variables
]regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
_trainable_variables
 ?layer_regularization_losses
?metrics
?layer_metrics
?non_trainable_variables
`	variables
aregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
/:-??2block5_conv1/kernel
 :?2block5_conv1/bias
.
c0
d1"
trackable_list_wrapper
.
c0
d1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
etrainable_variables
 ?layer_regularization_losses
?metrics
?layer_metrics
?non_trainable_variables
f	variables
gregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?


vkernel
wbias
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_first", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-3": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512, 7, 7]}}
?
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_first"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_first"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}
?

xkernel
ybias
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 50, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 144}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 144]}}
?

zkernel
{bias
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50]}}
"
	optimizer
-
?serving_default"
signature_map
 "
trackable_dict_wrapper
J
v0
w1
x2
y3
z4
{5"
trackable_list_wrapper
J
v0
w1
x2
y3
z4
{5"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
rtrainable_variables
 ?layer_regularization_losses
?metrics
?layer_metrics
?non_trainable_variables
s	variables
tregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
(:&?2conv2d/kernel
:2conv2d/bias
:	?22dense/kernel
:22
dense/bias
 :22dense_1/kernel
:2dense_1/bias
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
v0
w1"
trackable_list_wrapper
.
v0
w1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?trainable_variables
 ?layer_regularization_losses
?metrics
?layer_metrics
?non_trainable_variables
?	variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?trainable_variables
 ?layer_regularization_losses
?metrics
?layer_metrics
?non_trainable_variables
?	variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?trainable_variables
 ?layer_regularization_losses
?metrics
?layer_metrics
?non_trainable_variables
?	variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?trainable_variables
 ?layer_regularization_losses
?metrics
?layer_metrics
?non_trainable_variables
?	variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
.
x0
y1"
trackable_list_wrapper
.
x0
y1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?trainable_variables
 ?layer_regularization_losses
?metrics
?layer_metrics
?non_trainable_variables
?	variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
.
z0
{1"
trackable_list_wrapper
.
z0
{1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?trainable_variables
 ?layer_regularization_losses
?metrics
?layer_metrics
?non_trainable_variables
?	variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
J
i0
j1
k2
l3
m4
n5"
trackable_list_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "binary_accuracy"}}
:  (2total_1
:  (2count_1
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
?2?
__inference__wrapped_model_1310?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *.?+
)?&
input_1?????????xx
?2?
&__inference_model_1_layer_call_fn_2178
&__inference_model_1_layer_call_fn_2474
&__inference_model_1_layer_call_fn_2097
&__inference_model_1_layer_call_fn_2507?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
A__inference_model_1_layer_call_and_return_conditional_losses_2441
A__inference_model_1_layer_call_and_return_conditional_losses_2015
A__inference_model_1_layer_call_and_return_conditional_losses_1967
A__inference_model_1_layer_call_and_return_conditional_losses_2329?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_block1_conv1_layer_call_fn_2525?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_block1_conv1_layer_call_and_return_conditional_losses_2518?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_block1_conv2_layer_call_fn_2543?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_block1_conv2_layer_call_and_return_conditional_losses_2536?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_block1_pool_layer_call_fn_1327?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
E__inference_block1_pool_layer_call_and_return_conditional_losses_1316?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
+__inference_block2_conv1_layer_call_fn_2561?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_block2_conv1_layer_call_and_return_conditional_losses_2554?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_block2_conv2_layer_call_fn_2579?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_block2_conv2_layer_call_and_return_conditional_losses_2572?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_block2_pool_layer_call_fn_1344?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
E__inference_block2_pool_layer_call_and_return_conditional_losses_1333?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
+__inference_block3_conv1_layer_call_fn_2597?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_block3_conv1_layer_call_and_return_conditional_losses_2590?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_block3_conv2_layer_call_fn_2615?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_block3_conv2_layer_call_and_return_conditional_losses_2608?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_block3_conv3_layer_call_fn_2633?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_block3_conv3_layer_call_and_return_conditional_losses_2626?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_block3_pool_layer_call_fn_1361?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
E__inference_block3_pool_layer_call_and_return_conditional_losses_1350?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
+__inference_block4_conv1_layer_call_fn_2651?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_block4_conv1_layer_call_and_return_conditional_losses_2644?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_block4_conv2_layer_call_fn_2669?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_block4_conv2_layer_call_and_return_conditional_losses_2662?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_block4_conv3_layer_call_fn_2687?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_block4_conv3_layer_call_and_return_conditional_losses_2680?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_block4_pool_layer_call_fn_1378?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
E__inference_block4_pool_layer_call_and_return_conditional_losses_1367?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
+__inference_block5_conv1_layer_call_fn_2705?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_block5_conv1_layer_call_and_return_conditional_losses_2698?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_sequential_layer_call_fn_2782
)__inference_sequential_layer_call_fn_2881
)__inference_sequential_layer_call_fn_1590
)__inference_sequential_layer_call_fn_2793
)__inference_sequential_layer_call_fn_2870
)__inference_sequential_layer_call_fn_1563?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_sequential_layer_call_and_return_conditional_losses_1535
D__inference_sequential_layer_call_and_return_conditional_losses_2828
D__inference_sequential_layer_call_and_return_conditional_losses_2740
D__inference_sequential_layer_call_and_return_conditional_losses_2859
D__inference_sequential_layer_call_and_return_conditional_losses_2771
D__inference_sequential_layer_call_and_return_conditional_losses_1519?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
"__inference_signature_wrapper_2213input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
%__inference_conv2d_layer_call_fn_2899?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
@__inference_conv2d_layer_call_and_return_conditional_losses_2892?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_max_pooling2d_layer_call_fn_1395?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_1384?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
&__inference_flatten_layer_call_fn_2912?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_flatten_layer_call_and_return_conditional_losses_2907?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_dropout_layer_call_fn_2931
&__inference_dropout_layer_call_fn_2936?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
A__inference_dropout_layer_call_and_return_conditional_losses_2921
A__inference_dropout_layer_call_and_return_conditional_losses_2926?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
$__inference_dense_layer_call_fn_2954?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
?__inference_dense_layer_call_and_return_conditional_losses_2947?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_dense_1_layer_call_fn_2972?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_dense_1_layer_call_and_return_conditional_losses_2965?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
!__inference_signature_wrapper_434input_2"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
__inference__wrapped_model_1310?'(-.78=>CDMNSTYZcdvwxyz{8?5
.?+
)?&
input_1?????????xx
? "7?4
2

sequential$?!

sequential??????????
F__inference_block1_conv1_layer_call_and_return_conditional_losses_2518l7?4
-?*
(?%
inputs?????????xx
? "-?*
#? 
0?????????@xx
? ?
+__inference_block1_conv1_layer_call_fn_2525_7?4
-?*
(?%
inputs?????????xx
? " ??????????@xx?
F__inference_block1_conv2_layer_call_and_return_conditional_losses_2536l7?4
-?*
(?%
inputs?????????@xx
? "-?*
#? 
0?????????@xx
? ?
+__inference_block1_conv2_layer_call_fn_2543_7?4
-?*
(?%
inputs?????????@xx
? " ??????????@xx?
E__inference_block1_pool_layer_call_and_return_conditional_losses_1316?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
*__inference_block1_pool_layer_call_fn_1327?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
F__inference_block2_conv1_layer_call_and_return_conditional_losses_2554m'(7?4
-?*
(?%
inputs?????????@<<
? ".?+
$?!
0??????????<<
? ?
+__inference_block2_conv1_layer_call_fn_2561`'(7?4
-?*
(?%
inputs?????????@<<
? "!???????????<<?
F__inference_block2_conv2_layer_call_and_return_conditional_losses_2572n-.8?5
.?+
)?&
inputs??????????<<
? ".?+
$?!
0??????????<<
? ?
+__inference_block2_conv2_layer_call_fn_2579a-.8?5
.?+
)?&
inputs??????????<<
? "!???????????<<?
E__inference_block2_pool_layer_call_and_return_conditional_losses_1333?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
*__inference_block2_pool_layer_call_fn_1344?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
F__inference_block3_conv1_layer_call_and_return_conditional_losses_2590n788?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
+__inference_block3_conv1_layer_call_fn_2597a788?5
.?+
)?&
inputs??????????
? "!????????????
F__inference_block3_conv2_layer_call_and_return_conditional_losses_2608n=>8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
+__inference_block3_conv2_layer_call_fn_2615a=>8?5
.?+
)?&
inputs??????????
? "!????????????
F__inference_block3_conv3_layer_call_and_return_conditional_losses_2626nCD8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
+__inference_block3_conv3_layer_call_fn_2633aCD8?5
.?+
)?&
inputs??????????
? "!????????????
E__inference_block3_pool_layer_call_and_return_conditional_losses_1350?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
*__inference_block3_pool_layer_call_fn_1361?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
F__inference_block4_conv1_layer_call_and_return_conditional_losses_2644nMN8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
+__inference_block4_conv1_layer_call_fn_2651aMN8?5
.?+
)?&
inputs??????????
? "!????????????
F__inference_block4_conv2_layer_call_and_return_conditional_losses_2662nST8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
+__inference_block4_conv2_layer_call_fn_2669aST8?5
.?+
)?&
inputs??????????
? "!????????????
F__inference_block4_conv3_layer_call_and_return_conditional_losses_2680nYZ8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
+__inference_block4_conv3_layer_call_fn_2687aYZ8?5
.?+
)?&
inputs??????????
? "!????????????
E__inference_block4_pool_layer_call_and_return_conditional_losses_1367?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
*__inference_block4_pool_layer_call_fn_1378?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
F__inference_block5_conv1_layer_call_and_return_conditional_losses_2698ncd8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
+__inference_block5_conv1_layer_call_fn_2705acd8?5
.?+
)?&
inputs??????????
? "!????????????
@__inference_conv2d_layer_call_and_return_conditional_losses_2892mvw8?5
.?+
)?&
inputs??????????
? "-?*
#? 
0?????????
? ?
%__inference_conv2d_layer_call_fn_2899`vw8?5
.?+
)?&
inputs??????????
? " ???????????
A__inference_dense_1_layer_call_and_return_conditional_losses_2965\z{/?,
%?"
 ?
inputs?????????2
? "%?"
?
0?????????
? y
&__inference_dense_1_layer_call_fn_2972Oz{/?,
%?"
 ?
inputs?????????2
? "???????????
?__inference_dense_layer_call_and_return_conditional_losses_2947]xy0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????2
? x
$__inference_dense_layer_call_fn_2954Pxy0?-
&?#
!?
inputs??????????
? "??????????2?
A__inference_dropout_layer_call_and_return_conditional_losses_2921^4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? ?
A__inference_dropout_layer_call_and_return_conditional_losses_2926^4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? {
&__inference_dropout_layer_call_fn_2931Q4?1
*?'
!?
inputs??????????
p
? "???????????{
&__inference_dropout_layer_call_fn_2936Q4?1
*?'
!?
inputs??????????
p 
? "????????????
A__inference_flatten_layer_call_and_return_conditional_losses_2907a7?4
-?*
(?%
inputs?????????
? "&?#
?
0??????????
? ~
&__inference_flatten_layer_call_fn_2912T7?4
-?*
(?%
inputs?????????
? "????????????
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_1384?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
,__inference_max_pooling2d_layer_call_fn_1395?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
A__inference_model_1_layer_call_and_return_conditional_losses_1967?'(-.78=>CDMNSTYZcdvwxyz{@?=
6?3
)?&
input_1?????????xx
p

 
? "%?"
?
0?????????
? ?
A__inference_model_1_layer_call_and_return_conditional_losses_2015?'(-.78=>CDMNSTYZcdvwxyz{@?=
6?3
)?&
input_1?????????xx
p 

 
? "%?"
?
0?????????
? ?
A__inference_model_1_layer_call_and_return_conditional_losses_2329?'(-.78=>CDMNSTYZcdvwxyz{??<
5?2
(?%
inputs?????????xx
p

 
? "%?"
?
0?????????
? ?
A__inference_model_1_layer_call_and_return_conditional_losses_2441?'(-.78=>CDMNSTYZcdvwxyz{??<
5?2
(?%
inputs?????????xx
p 

 
? "%?"
?
0?????????
? ?
&__inference_model_1_layer_call_fn_2097z'(-.78=>CDMNSTYZcdvwxyz{@?=
6?3
)?&
input_1?????????xx
p

 
? "???????????
&__inference_model_1_layer_call_fn_2178z'(-.78=>CDMNSTYZcdvwxyz{@?=
6?3
)?&
input_1?????????xx
p 

 
? "???????????
&__inference_model_1_layer_call_fn_2474y'(-.78=>CDMNSTYZcdvwxyz{??<
5?2
(?%
inputs?????????xx
p

 
? "???????????
&__inference_model_1_layer_call_fn_2507y'(-.78=>CDMNSTYZcdvwxyz{??<
5?2
(?%
inputs?????????xx
p 

 
? "???????????
D__inference_sequential_layer_call_and_return_conditional_losses_1519rvwxyz{A?>
7?4
*?'
input_2??????????
p

 
? "%?"
?
0?????????
? ?
D__inference_sequential_layer_call_and_return_conditional_losses_1535rvwxyz{A?>
7?4
*?'
input_2??????????
p 

 
? "%?"
?
0?????????
? ?
D__inference_sequential_layer_call_and_return_conditional_losses_2740xvwxyz{G?D
=?:
0?-
+?(
inputs/0??????????
p

 
? "%?"
?
0?????????
? ?
D__inference_sequential_layer_call_and_return_conditional_losses_2771xvwxyz{G?D
=?:
0?-
+?(
inputs/0??????????
p 

 
? "%?"
?
0?????????
? ?
D__inference_sequential_layer_call_and_return_conditional_losses_2828qvwxyz{@?=
6?3
)?&
inputs??????????
p

 
? "%?"
?
0?????????
? ?
D__inference_sequential_layer_call_and_return_conditional_losses_2859qvwxyz{@?=
6?3
)?&
inputs??????????
p 

 
? "%?"
?
0?????????
? ?
)__inference_sequential_layer_call_fn_1563evwxyz{A?>
7?4
*?'
input_2??????????
p

 
? "???????????
)__inference_sequential_layer_call_fn_1590evwxyz{A?>
7?4
*?'
input_2??????????
p 

 
? "???????????
)__inference_sequential_layer_call_fn_2782kvwxyz{G?D
=?:
0?-
+?(
inputs/0??????????
p

 
? "???????????
)__inference_sequential_layer_call_fn_2793kvwxyz{G?D
=?:
0?-
+?(
inputs/0??????????
p 

 
? "???????????
)__inference_sequential_layer_call_fn_2870dvwxyz{@?=
6?3
)?&
inputs??????????
p

 
? "???????????
)__inference_sequential_layer_call_fn_2881dvwxyz{@?=
6?3
)?&
inputs??????????
p 

 
? "???????????
"__inference_signature_wrapper_2213?'(-.78=>CDMNSTYZcdvwxyz{C?@
? 
9?6
4
input_1)?&
input_1?????????xx"7?4
2

sequential$?!

sequential??????????
!__inference_signature_wrapper_434?vwxyz{D?A
? 
:?7
5
input_2*?'
input_2??????????"1?.
,
dense_1!?
dense_1?????????