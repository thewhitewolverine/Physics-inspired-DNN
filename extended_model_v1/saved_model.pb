��
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
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
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
?
Mul
x"T
y"T
z"T"
Ttype:
2	�
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
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
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
�
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
executor_typestring ��
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
<
Sub
x"T
y"T
z"T"
Ttype:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68��
z
dense_18/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Rf* 
shared_namedense_18/kernel
s
#dense_18/kernel/Read/ReadVariableOpReadVariableOpdense_18/kernel*
_output_shapes

:Rf*
dtype0
r
dense_18/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:f*
shared_namedense_18/bias
k
!dense_18/bias/Read/ReadVariableOpReadVariableOpdense_18/bias*
_output_shapes
:f*
dtype0
z
dense_19/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:ff* 
shared_namedense_19/kernel
s
#dense_19/kernel/Read/ReadVariableOpReadVariableOpdense_19/kernel*
_output_shapes

:ff*
dtype0
r
dense_19/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:f*
shared_namedense_19/bias
k
!dense_19/bias/Read/ReadVariableOpReadVariableOpdense_19/bias*
_output_shapes
:f*
dtype0
z
dense_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:ff* 
shared_namedense_20/kernel
s
#dense_20/kernel/Read/ReadVariableOpReadVariableOpdense_20/kernel*
_output_shapes

:ff*
dtype0
r
dense_20/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:f*
shared_namedense_20/bias
k
!dense_20/bias/Read/ReadVariableOpReadVariableOpdense_20/bias*
_output_shapes
:f*
dtype0
{
dense_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	f�* 
shared_namedense_21/kernel
t
#dense_21/kernel/Read/ReadVariableOpReadVariableOpdense_21/kernel*
_output_shapes
:	f�*
dtype0
s
dense_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_21/bias
l
!dense_21/bias/Read/ReadVariableOpReadVariableOpdense_21/bias*
_output_shapes	
:�*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
�
,module_wrapper_4/batch_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:f*=
shared_name.,module_wrapper_4/batch_normalization_4/gamma
�
@module_wrapper_4/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOp,module_wrapper_4/batch_normalization_4/gamma*
_output_shapes
:f*
dtype0
�
+module_wrapper_4/batch_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:f*<
shared_name-+module_wrapper_4/batch_normalization_4/beta
�
?module_wrapper_4/batch_normalization_4/beta/Read/ReadVariableOpReadVariableOp+module_wrapper_4/batch_normalization_4/beta*
_output_shapes
:f*
dtype0
�
2module_wrapper_4/batch_normalization_4/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:f*C
shared_name42module_wrapper_4/batch_normalization_4/moving_mean
�
Fmodule_wrapper_4/batch_normalization_4/moving_mean/Read/ReadVariableOpReadVariableOp2module_wrapper_4/batch_normalization_4/moving_mean*
_output_shapes
:f*
dtype0
�
6module_wrapper_4/batch_normalization_4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:f*G
shared_name86module_wrapper_4/batch_normalization_4/moving_variance
�
Jmodule_wrapper_4/batch_normalization_4/moving_variance/Read/ReadVariableOpReadVariableOp6module_wrapper_4/batch_normalization_4/moving_variance*
_output_shapes
:f*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
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
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
�
Adam/dense_18/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Rf*'
shared_nameAdam/dense_18/kernel/m
�
*Adam/dense_18/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_18/kernel/m*
_output_shapes

:Rf*
dtype0
�
Adam/dense_18/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:f*%
shared_nameAdam/dense_18/bias/m
y
(Adam/dense_18/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_18/bias/m*
_output_shapes
:f*
dtype0
�
Adam/dense_19/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:ff*'
shared_nameAdam/dense_19/kernel/m
�
*Adam/dense_19/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_19/kernel/m*
_output_shapes

:ff*
dtype0
�
Adam/dense_19/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:f*%
shared_nameAdam/dense_19/bias/m
y
(Adam/dense_19/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_19/bias/m*
_output_shapes
:f*
dtype0
�
Adam/dense_20/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:ff*'
shared_nameAdam/dense_20/kernel/m
�
*Adam/dense_20/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_20/kernel/m*
_output_shapes

:ff*
dtype0
�
Adam/dense_20/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:f*%
shared_nameAdam/dense_20/bias/m
y
(Adam/dense_20/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_20/bias/m*
_output_shapes
:f*
dtype0
�
Adam/dense_21/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	f�*'
shared_nameAdam/dense_21/kernel/m
�
*Adam/dense_21/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_21/kernel/m*
_output_shapes
:	f�*
dtype0
�
Adam/dense_21/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/dense_21/bias/m
z
(Adam/dense_21/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_21/bias/m*
_output_shapes	
:�*
dtype0
�
3Adam/module_wrapper_4/batch_normalization_4/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:f*D
shared_name53Adam/module_wrapper_4/batch_normalization_4/gamma/m
�
GAdam/module_wrapper_4/batch_normalization_4/gamma/m/Read/ReadVariableOpReadVariableOp3Adam/module_wrapper_4/batch_normalization_4/gamma/m*
_output_shapes
:f*
dtype0
�
2Adam/module_wrapper_4/batch_normalization_4/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:f*C
shared_name42Adam/module_wrapper_4/batch_normalization_4/beta/m
�
FAdam/module_wrapper_4/batch_normalization_4/beta/m/Read/ReadVariableOpReadVariableOp2Adam/module_wrapper_4/batch_normalization_4/beta/m*
_output_shapes
:f*
dtype0
�
Adam/dense_18/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Rf*'
shared_nameAdam/dense_18/kernel/v
�
*Adam/dense_18/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_18/kernel/v*
_output_shapes

:Rf*
dtype0
�
Adam/dense_18/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:f*%
shared_nameAdam/dense_18/bias/v
y
(Adam/dense_18/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_18/bias/v*
_output_shapes
:f*
dtype0
�
Adam/dense_19/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:ff*'
shared_nameAdam/dense_19/kernel/v
�
*Adam/dense_19/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_19/kernel/v*
_output_shapes

:ff*
dtype0
�
Adam/dense_19/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:f*%
shared_nameAdam/dense_19/bias/v
y
(Adam/dense_19/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_19/bias/v*
_output_shapes
:f*
dtype0
�
Adam/dense_20/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:ff*'
shared_nameAdam/dense_20/kernel/v
�
*Adam/dense_20/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_20/kernel/v*
_output_shapes

:ff*
dtype0
�
Adam/dense_20/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:f*%
shared_nameAdam/dense_20/bias/v
y
(Adam/dense_20/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_20/bias/v*
_output_shapes
:f*
dtype0
�
Adam/dense_21/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	f�*'
shared_nameAdam/dense_21/kernel/v
�
*Adam/dense_21/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_21/kernel/v*
_output_shapes
:	f�*
dtype0
�
Adam/dense_21/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/dense_21/bias/v
z
(Adam/dense_21/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_21/bias/v*
_output_shapes	
:�*
dtype0
�
3Adam/module_wrapper_4/batch_normalization_4/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:f*D
shared_name53Adam/module_wrapper_4/batch_normalization_4/gamma/v
�
GAdam/module_wrapper_4/batch_normalization_4/gamma/v/Read/ReadVariableOpReadVariableOp3Adam/module_wrapper_4/batch_normalization_4/gamma/v*
_output_shapes
:f*
dtype0
�
2Adam/module_wrapper_4/batch_normalization_4/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:f*C
shared_name42Adam/module_wrapper_4/batch_normalization_4/beta/v
�
FAdam/module_wrapper_4/batch_normalization_4/beta/v/Read/ReadVariableOpReadVariableOp2Adam/module_wrapper_4/batch_normalization_4/beta/v*
_output_shapes
:f*
dtype0

NoOpNoOp
�P
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�P
value�OB�O B�O
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer-4
layer_with_weights-4
layer-5
	optimizer
trainable_variables
	regularization_losses

	variables
	keras_api
_default_save_signature
__call__
*&call_and_return_all_conditional_losses

signatures*
�

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
__call__
*&call_and_return_all_conditional_losses*
�

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
__call__
*&call_and_return_all_conditional_losses*
�

 kernel
!bias
"trainable_variables
#regularization_losses
$	variables
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses*
�
(_module
)trainable_variables
*regularization_losses
+	variables
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses*
�
/trainable_variables
0regularization_losses
1	variables
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses* 
�

5kernel
6bias
7trainable_variables
8regularization_losses
9	variables
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses*
�
=iter

>beta_1

?beta_2
	@decay
Alearning_ratem�m�m�m� m�!m�5m�6m�Bm�Cm�v�v�v�v� v�!v�5v�6v�Bv�Cv�*
J
0
1
2
3
 4
!5
B6
C7
58
69*
* 
Z
0
1
2
3
 4
!5
B6
C7
D8
E9
510
611*
�
Flayer_metrics
Gmetrics
trainable_variables

Hlayers
	regularization_losses

	variables
Ilayer_regularization_losses
Jnon_trainable_variables
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

Kserving_default* 
_Y
VARIABLE_VALUEdense_18/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_18/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*
* 

0
1*
�
Llayer_metrics
Mmetrics
trainable_variables

Nlayers
regularization_losses
	variables
Olayer_regularization_losses
Pnon_trainable_variables
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
_Y
VARIABLE_VALUEdense_19/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_19/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*
* 

0
1*
�
Qlayer_metrics
Rmetrics
trainable_variables

Slayers
regularization_losses
	variables
Tlayer_regularization_losses
Unon_trainable_variables
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
_Y
VARIABLE_VALUEdense_20/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_20/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

 0
!1*
* 

 0
!1*
�
Vlayer_metrics
Wmetrics
"trainable_variables

Xlayers
#regularization_losses
$	variables
Ylayer_regularization_losses
Znon_trainable_variables
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses*
* 
* 
�
[axis
	Bgamma
Cbeta
Dmoving_mean
Emoving_variance
\	variables
]trainable_variables
^regularization_losses
_	keras_api
`__call__
*a&call_and_return_all_conditional_losses*

B0
C1*
* 
 
B0
C1
D2
E3*
�
blayer_metrics
cmetrics
)trainable_variables

dlayers
*regularization_losses
+	variables
elayer_regularization_losses
fnon_trainable_variables
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses*
* 
* 
* 
* 
* 
�
glayer_metrics
hmetrics
/trainable_variables

ilayers
0regularization_losses
1	variables
jlayer_regularization_losses
knon_trainable_variables
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses* 
* 
* 
_Y
VARIABLE_VALUEdense_21/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_21/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*

50
61*
* 

50
61*
�
llayer_metrics
mmetrics
7trainable_variables

nlayers
8regularization_losses
9	variables
olayer_regularization_losses
pnon_trainable_variables
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses*
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE,module_wrapper_4/batch_normalization_4/gamma0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE+module_wrapper_4/batch_normalization_4/beta0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE2module_wrapper_4/batch_normalization_4/moving_mean&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE6module_wrapper_4/batch_normalization_4/moving_variance&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
* 

q0
r1*
.
0
1
2
3
4
5*
* 

D0
E1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
 
B0
C1
D2
E3*

B0
C1*
* 
�
snon_trainable_variables

tlayers
umetrics
vlayer_regularization_losses
wlayer_metrics
\	variables
]trainable_variables
^regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 

D0
E1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
8
	xtotal
	ycount
z	variables
{	keras_api*
S
	|total
	}count
~_fn

_fn_kwargs
�	variables
�	keras_api*

D0
E1*
* 
* 
* 
* 
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

x0
y1*

z	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
M

�total

�count
�
_fn_kwargs
�	variables
�	keras_api*
* 

|0
}1*

�	variables*
YS
VARIABLE_VALUEtotal_28keras_api/metrics/1/_fn/total/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEcount_28keras_api/metrics/1/_fn/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

�0
�1*

�	variables*
�|
VARIABLE_VALUEAdam/dense_18/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_18/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/dense_19/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_19/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/dense_20/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_20/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/dense_21/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_21/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE3Adam/module_wrapper_4/batch_normalization_4/gamma/mLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE2Adam/module_wrapper_4/batch_normalization_4/beta/mLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/dense_18/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_18/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/dense_19/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_19/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/dense_20/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_20/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/dense_21/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_21/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE3Adam/module_wrapper_4/batch_normalization_4/gamma/vLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE2Adam/module_wrapper_4/batch_normalization_4/beta/vLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�
serving_default_dense_18_inputPlaceholder*'
_output_shapes
:���������R*
dtype0*
shape:���������R
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_dense_18_inputdense_18/kerneldense_18/biasdense_19/kerneldense_19/biasdense_20/kerneldense_20/bias6module_wrapper_4/batch_normalization_4/moving_variance,module_wrapper_4/batch_normalization_4/gamma2module_wrapper_4/batch_normalization_4/moving_mean+module_wrapper_4/batch_normalization_4/betadense_21/kerneldense_21/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� */
f*R(
&__inference_signature_wrapper_36603689
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
z
StaticRegexFullMatchStaticRegexFullMatchsaver_filename"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*
\
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part
a
Const_2Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
h
SelectSelectStaticRegexFullMatchConst_1Const_2"/device:CPU:**
T0*
_output_shapes
: 
`

StringJoin
StringJoinsaver_filenameSelect"/device:CPU:**
N*
_output_shapes
: 
L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :
f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
x
ShardedFilenameShardedFilename
StringJoinShardedFilename/shard
num_shards"/device:CPU:0*
_output_shapes
: 
�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*�
value�B�,B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB8keras_api/metrics/1/_fn/total/.ATTRIBUTES/VARIABLE_VALUEB8keras_api/metrics/1/_fn/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*k
valuebB`,B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
�
SaveV2SaveV2ShardedFilenameSaveV2/tensor_namesSaveV2/shape_and_slices#dense_18/kernel/Read/ReadVariableOp!dense_18/bias/Read/ReadVariableOp#dense_19/kernel/Read/ReadVariableOp!dense_19/bias/Read/ReadVariableOp#dense_20/kernel/Read/ReadVariableOp!dense_20/bias/Read/ReadVariableOp#dense_21/kernel/Read/ReadVariableOp!dense_21/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp@module_wrapper_4/batch_normalization_4/gamma/Read/ReadVariableOp?module_wrapper_4/batch_normalization_4/beta/Read/ReadVariableOpFmodule_wrapper_4/batch_normalization_4/moving_mean/Read/ReadVariableOpJmodule_wrapper_4/batch_normalization_4/moving_variance/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOp*Adam/dense_18/kernel/m/Read/ReadVariableOp(Adam/dense_18/bias/m/Read/ReadVariableOp*Adam/dense_19/kernel/m/Read/ReadVariableOp(Adam/dense_19/bias/m/Read/ReadVariableOp*Adam/dense_20/kernel/m/Read/ReadVariableOp(Adam/dense_20/bias/m/Read/ReadVariableOp*Adam/dense_21/kernel/m/Read/ReadVariableOp(Adam/dense_21/bias/m/Read/ReadVariableOpGAdam/module_wrapper_4/batch_normalization_4/gamma/m/Read/ReadVariableOpFAdam/module_wrapper_4/batch_normalization_4/beta/m/Read/ReadVariableOp*Adam/dense_18/kernel/v/Read/ReadVariableOp(Adam/dense_18/bias/v/Read/ReadVariableOp*Adam/dense_19/kernel/v/Read/ReadVariableOp(Adam/dense_19/bias/v/Read/ReadVariableOp*Adam/dense_20/kernel/v/Read/ReadVariableOp(Adam/dense_20/bias/v/Read/ReadVariableOp*Adam/dense_21/kernel/v/Read/ReadVariableOp(Adam/dense_21/bias/v/Read/ReadVariableOpGAdam/module_wrapper_4/batch_normalization_4/gamma/v/Read/ReadVariableOpFAdam/module_wrapper_4/batch_normalization_4/beta/v/Read/ReadVariableOpConst"/device:CPU:0*:
dtypes0
.2,	
�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
o
MergeV2CheckpointsMergeV2Checkpoints&MergeV2Checkpoints/checkpoint_prefixessaver_filename"/device:CPU:0
i
IdentityIdentitysaver_filename^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 
�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*�
value�B�,B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB8keras_api/metrics/1/_fn/total/.ATTRIBUTES/VARIABLE_VALUEB8keras_api/metrics/1/_fn/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:,*
dtype0*k
valuebB`,B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
�
	RestoreV2	RestoreV2saver_filenameRestoreV2/tensor_namesRestoreV2/shape_and_slices"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::*:
dtypes0
.2,	
S

Identity_1Identity	RestoreV2"/device:CPU:0*
T0*
_output_shapes
:
]
AssignVariableOpAssignVariableOpdense_18/kernel
Identity_1"/device:CPU:0*
dtype0
U

Identity_2IdentityRestoreV2:1"/device:CPU:0*
T0*
_output_shapes
:
]
AssignVariableOp_1AssignVariableOpdense_18/bias
Identity_2"/device:CPU:0*
dtype0
U

Identity_3IdentityRestoreV2:2"/device:CPU:0*
T0*
_output_shapes
:
_
AssignVariableOp_2AssignVariableOpdense_19/kernel
Identity_3"/device:CPU:0*
dtype0
U

Identity_4IdentityRestoreV2:3"/device:CPU:0*
T0*
_output_shapes
:
]
AssignVariableOp_3AssignVariableOpdense_19/bias
Identity_4"/device:CPU:0*
dtype0
U

Identity_5IdentityRestoreV2:4"/device:CPU:0*
T0*
_output_shapes
:
_
AssignVariableOp_4AssignVariableOpdense_20/kernel
Identity_5"/device:CPU:0*
dtype0
U

Identity_6IdentityRestoreV2:5"/device:CPU:0*
T0*
_output_shapes
:
]
AssignVariableOp_5AssignVariableOpdense_20/bias
Identity_6"/device:CPU:0*
dtype0
U

Identity_7IdentityRestoreV2:6"/device:CPU:0*
T0*
_output_shapes
:
_
AssignVariableOp_6AssignVariableOpdense_21/kernel
Identity_7"/device:CPU:0*
dtype0
U

Identity_8IdentityRestoreV2:7"/device:CPU:0*
T0*
_output_shapes
:
]
AssignVariableOp_7AssignVariableOpdense_21/bias
Identity_8"/device:CPU:0*
dtype0
U

Identity_9IdentityRestoreV2:8"/device:CPU:0*
T0	*
_output_shapes
:
Y
AssignVariableOp_8AssignVariableOp	Adam/iter
Identity_9"/device:CPU:0*
dtype0	
V
Identity_10IdentityRestoreV2:9"/device:CPU:0*
T0*
_output_shapes
:
\
AssignVariableOp_9AssignVariableOpAdam/beta_1Identity_10"/device:CPU:0*
dtype0
W
Identity_11IdentityRestoreV2:10"/device:CPU:0*
T0*
_output_shapes
:
]
AssignVariableOp_10AssignVariableOpAdam/beta_2Identity_11"/device:CPU:0*
dtype0
W
Identity_12IdentityRestoreV2:11"/device:CPU:0*
T0*
_output_shapes
:
\
AssignVariableOp_11AssignVariableOp
Adam/decayIdentity_12"/device:CPU:0*
dtype0
W
Identity_13IdentityRestoreV2:12"/device:CPU:0*
T0*
_output_shapes
:
d
AssignVariableOp_12AssignVariableOpAdam/learning_rateIdentity_13"/device:CPU:0*
dtype0
W
Identity_14IdentityRestoreV2:13"/device:CPU:0*
T0*
_output_shapes
:
~
AssignVariableOp_13AssignVariableOp,module_wrapper_4/batch_normalization_4/gammaIdentity_14"/device:CPU:0*
dtype0
W
Identity_15IdentityRestoreV2:14"/device:CPU:0*
T0*
_output_shapes
:
}
AssignVariableOp_14AssignVariableOp+module_wrapper_4/batch_normalization_4/betaIdentity_15"/device:CPU:0*
dtype0
W
Identity_16IdentityRestoreV2:15"/device:CPU:0*
T0*
_output_shapes
:
�
AssignVariableOp_15AssignVariableOp2module_wrapper_4/batch_normalization_4/moving_meanIdentity_16"/device:CPU:0*
dtype0
W
Identity_17IdentityRestoreV2:16"/device:CPU:0*
T0*
_output_shapes
:
�
AssignVariableOp_16AssignVariableOp6module_wrapper_4/batch_normalization_4/moving_varianceIdentity_17"/device:CPU:0*
dtype0
W
Identity_18IdentityRestoreV2:17"/device:CPU:0*
T0*
_output_shapes
:
W
AssignVariableOp_17AssignVariableOptotalIdentity_18"/device:CPU:0*
dtype0
W
Identity_19IdentityRestoreV2:18"/device:CPU:0*
T0*
_output_shapes
:
W
AssignVariableOp_18AssignVariableOpcountIdentity_19"/device:CPU:0*
dtype0
W
Identity_20IdentityRestoreV2:19"/device:CPU:0*
T0*
_output_shapes
:
Y
AssignVariableOp_19AssignVariableOptotal_1Identity_20"/device:CPU:0*
dtype0
W
Identity_21IdentityRestoreV2:20"/device:CPU:0*
T0*
_output_shapes
:
Y
AssignVariableOp_20AssignVariableOpcount_1Identity_21"/device:CPU:0*
dtype0
W
Identity_22IdentityRestoreV2:21"/device:CPU:0*
T0*
_output_shapes
:
Y
AssignVariableOp_21AssignVariableOptotal_2Identity_22"/device:CPU:0*
dtype0
W
Identity_23IdentityRestoreV2:22"/device:CPU:0*
T0*
_output_shapes
:
Y
AssignVariableOp_22AssignVariableOpcount_2Identity_23"/device:CPU:0*
dtype0
W
Identity_24IdentityRestoreV2:23"/device:CPU:0*
T0*
_output_shapes
:
h
AssignVariableOp_23AssignVariableOpAdam/dense_18/kernel/mIdentity_24"/device:CPU:0*
dtype0
W
Identity_25IdentityRestoreV2:24"/device:CPU:0*
T0*
_output_shapes
:
f
AssignVariableOp_24AssignVariableOpAdam/dense_18/bias/mIdentity_25"/device:CPU:0*
dtype0
W
Identity_26IdentityRestoreV2:25"/device:CPU:0*
T0*
_output_shapes
:
h
AssignVariableOp_25AssignVariableOpAdam/dense_19/kernel/mIdentity_26"/device:CPU:0*
dtype0
W
Identity_27IdentityRestoreV2:26"/device:CPU:0*
T0*
_output_shapes
:
f
AssignVariableOp_26AssignVariableOpAdam/dense_19/bias/mIdentity_27"/device:CPU:0*
dtype0
W
Identity_28IdentityRestoreV2:27"/device:CPU:0*
T0*
_output_shapes
:
h
AssignVariableOp_27AssignVariableOpAdam/dense_20/kernel/mIdentity_28"/device:CPU:0*
dtype0
W
Identity_29IdentityRestoreV2:28"/device:CPU:0*
T0*
_output_shapes
:
f
AssignVariableOp_28AssignVariableOpAdam/dense_20/bias/mIdentity_29"/device:CPU:0*
dtype0
W
Identity_30IdentityRestoreV2:29"/device:CPU:0*
T0*
_output_shapes
:
h
AssignVariableOp_29AssignVariableOpAdam/dense_21/kernel/mIdentity_30"/device:CPU:0*
dtype0
W
Identity_31IdentityRestoreV2:30"/device:CPU:0*
T0*
_output_shapes
:
f
AssignVariableOp_30AssignVariableOpAdam/dense_21/bias/mIdentity_31"/device:CPU:0*
dtype0
W
Identity_32IdentityRestoreV2:31"/device:CPU:0*
T0*
_output_shapes
:
�
AssignVariableOp_31AssignVariableOp3Adam/module_wrapper_4/batch_normalization_4/gamma/mIdentity_32"/device:CPU:0*
dtype0
W
Identity_33IdentityRestoreV2:32"/device:CPU:0*
T0*
_output_shapes
:
�
AssignVariableOp_32AssignVariableOp2Adam/module_wrapper_4/batch_normalization_4/beta/mIdentity_33"/device:CPU:0*
dtype0
W
Identity_34IdentityRestoreV2:33"/device:CPU:0*
T0*
_output_shapes
:
h
AssignVariableOp_33AssignVariableOpAdam/dense_18/kernel/vIdentity_34"/device:CPU:0*
dtype0
W
Identity_35IdentityRestoreV2:34"/device:CPU:0*
T0*
_output_shapes
:
f
AssignVariableOp_34AssignVariableOpAdam/dense_18/bias/vIdentity_35"/device:CPU:0*
dtype0
W
Identity_36IdentityRestoreV2:35"/device:CPU:0*
T0*
_output_shapes
:
h
AssignVariableOp_35AssignVariableOpAdam/dense_19/kernel/vIdentity_36"/device:CPU:0*
dtype0
W
Identity_37IdentityRestoreV2:36"/device:CPU:0*
T0*
_output_shapes
:
f
AssignVariableOp_36AssignVariableOpAdam/dense_19/bias/vIdentity_37"/device:CPU:0*
dtype0
W
Identity_38IdentityRestoreV2:37"/device:CPU:0*
T0*
_output_shapes
:
h
AssignVariableOp_37AssignVariableOpAdam/dense_20/kernel/vIdentity_38"/device:CPU:0*
dtype0
W
Identity_39IdentityRestoreV2:38"/device:CPU:0*
T0*
_output_shapes
:
f
AssignVariableOp_38AssignVariableOpAdam/dense_20/bias/vIdentity_39"/device:CPU:0*
dtype0
W
Identity_40IdentityRestoreV2:39"/device:CPU:0*
T0*
_output_shapes
:
h
AssignVariableOp_39AssignVariableOpAdam/dense_21/kernel/vIdentity_40"/device:CPU:0*
dtype0
W
Identity_41IdentityRestoreV2:40"/device:CPU:0*
T0*
_output_shapes
:
f
AssignVariableOp_40AssignVariableOpAdam/dense_21/bias/vIdentity_41"/device:CPU:0*
dtype0
W
Identity_42IdentityRestoreV2:41"/device:CPU:0*
T0*
_output_shapes
:
�
AssignVariableOp_41AssignVariableOp3Adam/module_wrapper_4/batch_normalization_4/gamma/vIdentity_42"/device:CPU:0*
dtype0
W
Identity_43IdentityRestoreV2:42"/device:CPU:0*
T0*
_output_shapes
:
�
AssignVariableOp_42AssignVariableOp2Adam/module_wrapper_4/batch_normalization_4/beta/vIdentity_43"/device:CPU:0*
dtype0

NoOp_1NoOp"/device:CPU:0
�
Identity_44Identitysaver_filename^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp_1"/device:CPU:0*
T0*
_output_shapes
: ��
�

�
&__inference_signature_wrapper_36603689
dense_18_input
unknown:Rf
	unknown_0:f
	unknown_1:ff
	unknown_2:f
	unknown_3:ff
	unknown_4:f
	unknown_5:f
	unknown_6:f
	unknown_7:f
	unknown_8:f
	unknown_9:	f�

unknown_10:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCalldense_18_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *,
f'R%
#__inference__wrapped_model_36602772p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������R: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
'
_output_shapes
:���������R
(
_user_specified_namedense_18_input
�C
�
/__inference_sequential_4_layer_call_fn_36603488

inputs9
'dense_18_matmul_readvariableop_resource:Rf6
(dense_18_biasadd_readvariableop_resource:f9
'dense_19_matmul_readvariableop_resource:ff6
(dense_19_biasadd_readvariableop_resource:f9
'dense_20_matmul_readvariableop_resource:ff6
(dense_20_biasadd_readvariableop_resource:fV
Hmodule_wrapper_4_batch_normalization_4_batchnorm_readvariableop_resource:fZ
Lmodule_wrapper_4_batch_normalization_4_batchnorm_mul_readvariableop_resource:fX
Jmodule_wrapper_4_batch_normalization_4_batchnorm_readvariableop_1_resource:fX
Jmodule_wrapper_4_batch_normalization_4_batchnorm_readvariableop_2_resource:f:
'dense_21_matmul_readvariableop_resource:	f�7
(dense_21_biasadd_readvariableop_resource:	�
identity��dense_18/BiasAdd/ReadVariableOp�dense_18/MatMul/ReadVariableOp�dense_19/BiasAdd/ReadVariableOp�dense_19/MatMul/ReadVariableOp�dense_20/BiasAdd/ReadVariableOp�dense_20/MatMul/ReadVariableOp�dense_21/BiasAdd/ReadVariableOp�dense_21/MatMul/ReadVariableOp�?module_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp�Amodule_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp_1�Amodule_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp_2�Cmodule_wrapper_4/batch_normalization_4/batchnorm/mul/ReadVariableOp�
dense_18/MatMul/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource*
_output_shapes

:Rf*
dtype0{
dense_18/MatMulMatMulinputs&dense_18/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f�
dense_18/BiasAdd/ReadVariableOpReadVariableOp(dense_18_biasadd_readvariableop_resource*
_output_shapes
:f*
dtype0�
dense_18/BiasAddBiasAdddense_18/MatMul:product:0'dense_18/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������fb
dense_18/ReluReludense_18/BiasAdd:output:0*
T0*'
_output_shapes
:���������f�
dense_19/MatMul/ReadVariableOpReadVariableOp'dense_19_matmul_readvariableop_resource*
_output_shapes

:ff*
dtype0�
dense_19/MatMulMatMuldense_18/Relu:activations:0&dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f�
dense_19/BiasAdd/ReadVariableOpReadVariableOp(dense_19_biasadd_readvariableop_resource*
_output_shapes
:f*
dtype0�
dense_19/BiasAddBiasAdddense_19/MatMul:product:0'dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������fb
dense_19/ReluReludense_19/BiasAdd:output:0*
T0*'
_output_shapes
:���������f�
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource*
_output_shapes

:ff*
dtype0�
dense_20/MatMulMatMuldense_19/Relu:activations:0&dense_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f�
dense_20/BiasAdd/ReadVariableOpReadVariableOp(dense_20_biasadd_readvariableop_resource*
_output_shapes
:f*
dtype0�
dense_20/BiasAddBiasAdddense_20/MatMul:product:0'dense_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������fb
dense_20/ReluReludense_20/BiasAdd:output:0*
T0*'
_output_shapes
:���������f�
?module_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOpReadVariableOpHmodule_wrapper_4_batch_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
:f*
dtype0{
6module_wrapper_4/batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
4module_wrapper_4/batch_normalization_4/batchnorm/addAddV2Gmodule_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp:value:0?module_wrapper_4/batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes
:f�
6module_wrapper_4/batch_normalization_4/batchnorm/RsqrtRsqrt8module_wrapper_4/batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes
:f�
Cmodule_wrapper_4/batch_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOpLmodule_wrapper_4_batch_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
:f*
dtype0�
4module_wrapper_4/batch_normalization_4/batchnorm/mulMul:module_wrapper_4/batch_normalization_4/batchnorm/Rsqrt:y:0Kmodule_wrapper_4/batch_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:f�
6module_wrapper_4/batch_normalization_4/batchnorm/mul_1Muldense_20/Relu:activations:08module_wrapper_4/batch_normalization_4/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������f�
Amodule_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp_1ReadVariableOpJmodule_wrapper_4_batch_normalization_4_batchnorm_readvariableop_1_resource*
_output_shapes
:f*
dtype0�
6module_wrapper_4/batch_normalization_4/batchnorm/mul_2MulImodule_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp_1:value:08module_wrapper_4/batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes
:f�
Amodule_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp_2ReadVariableOpJmodule_wrapper_4_batch_normalization_4_batchnorm_readvariableop_2_resource*
_output_shapes
:f*
dtype0�
4module_wrapper_4/batch_normalization_4/batchnorm/subSubImodule_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp_2:value:0:module_wrapper_4/batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes
:f�
6module_wrapper_4/batch_normalization_4/batchnorm/add_1AddV2:module_wrapper_4/batch_normalization_4/batchnorm/mul_1:z:08module_wrapper_4/batch_normalization_4/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������f�
dropout_4/IdentityIdentity:module_wrapper_4/batch_normalization_4/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������f�
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource*
_output_shapes
:	f�*
dtype0�
dense_21/MatMulMatMuldropout_4/Identity:output:0&dense_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������i
IdentityIdentitydense_21/BiasAdd:output:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp@^module_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOpB^module_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp_1B^module_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp_2D^module_wrapper_4/batch_normalization_4/batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������R: : : : : : : : : : : : 2B
dense_18/BiasAdd/ReadVariableOpdense_18/BiasAdd/ReadVariableOp2@
dense_18/MatMul/ReadVariableOpdense_18/MatMul/ReadVariableOp2B
dense_19/BiasAdd/ReadVariableOpdense_19/BiasAdd/ReadVariableOp2@
dense_19/MatMul/ReadVariableOpdense_19/MatMul/ReadVariableOp2B
dense_20/BiasAdd/ReadVariableOpdense_20/BiasAdd/ReadVariableOp2@
dense_20/MatMul/ReadVariableOpdense_20/MatMul/ReadVariableOp2B
dense_21/BiasAdd/ReadVariableOpdense_21/BiasAdd/ReadVariableOp2@
dense_21/MatMul/ReadVariableOpdense_21/MatMul/ReadVariableOp2�
?module_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp?module_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp2�
Amodule_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp_1Amodule_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp_12�
Amodule_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp_2Amodule_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp_22�
Cmodule_wrapper_4/batch_normalization_4/batchnorm/mul/ReadVariableOpCmodule_wrapper_4/batch_normalization_4/batchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������R
 
_user_specified_nameinputs
�
�
8__inference_batch_normalization_4_layer_call_fn_36604031

inputs/
!batchnorm_readvariableop_resource:f3
%batchnorm_mul_readvariableop_resource:f1
#batchnorm_readvariableop_1_resource:f1
#batchnorm_readvariableop_2_resource:f
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:f*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:fP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:f~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:f*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:fc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������fz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:f*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:fz
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:f*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:fr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������fb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������f�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������f: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������f
 
_user_specified_nameinputs
�C
�
J__inference_sequential_4_layer_call_and_return_conditional_losses_36603597

inputs9
'dense_18_matmul_readvariableop_resource:Rf6
(dense_18_biasadd_readvariableop_resource:f9
'dense_19_matmul_readvariableop_resource:ff6
(dense_19_biasadd_readvariableop_resource:f9
'dense_20_matmul_readvariableop_resource:ff6
(dense_20_biasadd_readvariableop_resource:fV
Hmodule_wrapper_4_batch_normalization_4_batchnorm_readvariableop_resource:fZ
Lmodule_wrapper_4_batch_normalization_4_batchnorm_mul_readvariableop_resource:fX
Jmodule_wrapper_4_batch_normalization_4_batchnorm_readvariableop_1_resource:fX
Jmodule_wrapper_4_batch_normalization_4_batchnorm_readvariableop_2_resource:f:
'dense_21_matmul_readvariableop_resource:	f�7
(dense_21_biasadd_readvariableop_resource:	�
identity��dense_18/BiasAdd/ReadVariableOp�dense_18/MatMul/ReadVariableOp�dense_19/BiasAdd/ReadVariableOp�dense_19/MatMul/ReadVariableOp�dense_20/BiasAdd/ReadVariableOp�dense_20/MatMul/ReadVariableOp�dense_21/BiasAdd/ReadVariableOp�dense_21/MatMul/ReadVariableOp�?module_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp�Amodule_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp_1�Amodule_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp_2�Cmodule_wrapper_4/batch_normalization_4/batchnorm/mul/ReadVariableOp�
dense_18/MatMul/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource*
_output_shapes

:Rf*
dtype0{
dense_18/MatMulMatMulinputs&dense_18/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f�
dense_18/BiasAdd/ReadVariableOpReadVariableOp(dense_18_biasadd_readvariableop_resource*
_output_shapes
:f*
dtype0�
dense_18/BiasAddBiasAdddense_18/MatMul:product:0'dense_18/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������fb
dense_18/ReluReludense_18/BiasAdd:output:0*
T0*'
_output_shapes
:���������f�
dense_19/MatMul/ReadVariableOpReadVariableOp'dense_19_matmul_readvariableop_resource*
_output_shapes

:ff*
dtype0�
dense_19/MatMulMatMuldense_18/Relu:activations:0&dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f�
dense_19/BiasAdd/ReadVariableOpReadVariableOp(dense_19_biasadd_readvariableop_resource*
_output_shapes
:f*
dtype0�
dense_19/BiasAddBiasAdddense_19/MatMul:product:0'dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������fb
dense_19/ReluReludense_19/BiasAdd:output:0*
T0*'
_output_shapes
:���������f�
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource*
_output_shapes

:ff*
dtype0�
dense_20/MatMulMatMuldense_19/Relu:activations:0&dense_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f�
dense_20/BiasAdd/ReadVariableOpReadVariableOp(dense_20_biasadd_readvariableop_resource*
_output_shapes
:f*
dtype0�
dense_20/BiasAddBiasAdddense_20/MatMul:product:0'dense_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������fb
dense_20/ReluReludense_20/BiasAdd:output:0*
T0*'
_output_shapes
:���������f�
?module_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOpReadVariableOpHmodule_wrapper_4_batch_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
:f*
dtype0{
6module_wrapper_4/batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
4module_wrapper_4/batch_normalization_4/batchnorm/addAddV2Gmodule_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp:value:0?module_wrapper_4/batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes
:f�
6module_wrapper_4/batch_normalization_4/batchnorm/RsqrtRsqrt8module_wrapper_4/batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes
:f�
Cmodule_wrapper_4/batch_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOpLmodule_wrapper_4_batch_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
:f*
dtype0�
4module_wrapper_4/batch_normalization_4/batchnorm/mulMul:module_wrapper_4/batch_normalization_4/batchnorm/Rsqrt:y:0Kmodule_wrapper_4/batch_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:f�
6module_wrapper_4/batch_normalization_4/batchnorm/mul_1Muldense_20/Relu:activations:08module_wrapper_4/batch_normalization_4/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������f�
Amodule_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp_1ReadVariableOpJmodule_wrapper_4_batch_normalization_4_batchnorm_readvariableop_1_resource*
_output_shapes
:f*
dtype0�
6module_wrapper_4/batch_normalization_4/batchnorm/mul_2MulImodule_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp_1:value:08module_wrapper_4/batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes
:f�
Amodule_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp_2ReadVariableOpJmodule_wrapper_4_batch_normalization_4_batchnorm_readvariableop_2_resource*
_output_shapes
:f*
dtype0�
4module_wrapper_4/batch_normalization_4/batchnorm/subSubImodule_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp_2:value:0:module_wrapper_4/batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes
:f�
6module_wrapper_4/batch_normalization_4/batchnorm/add_1AddV2:module_wrapper_4/batch_normalization_4/batchnorm/mul_1:z:08module_wrapper_4/batch_normalization_4/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������f�
dropout_4/IdentityIdentity:module_wrapper_4/batch_normalization_4/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������f�
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource*
_output_shapes
:	f�*
dtype0�
dense_21/MatMulMatMuldropout_4/Identity:output:0&dense_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������i
IdentityIdentitydense_21/BiasAdd:output:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp@^module_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOpB^module_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp_1B^module_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp_2D^module_wrapper_4/batch_normalization_4/batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������R: : : : : : : : : : : : 2B
dense_18/BiasAdd/ReadVariableOpdense_18/BiasAdd/ReadVariableOp2@
dense_18/MatMul/ReadVariableOpdense_18/MatMul/ReadVariableOp2B
dense_19/BiasAdd/ReadVariableOpdense_19/BiasAdd/ReadVariableOp2@
dense_19/MatMul/ReadVariableOpdense_19/MatMul/ReadVariableOp2B
dense_20/BiasAdd/ReadVariableOpdense_20/BiasAdd/ReadVariableOp2@
dense_20/MatMul/ReadVariableOpdense_20/MatMul/ReadVariableOp2B
dense_21/BiasAdd/ReadVariableOpdense_21/BiasAdd/ReadVariableOp2@
dense_21/MatMul/ReadVariableOpdense_21/MatMul/ReadVariableOp2�
?module_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp?module_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp2�
Amodule_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp_1Amodule_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp_12�
Amodule_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp_2Amodule_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp_22�
Cmodule_wrapper_4/batch_normalization_4/batchnorm/mul/ReadVariableOpCmodule_wrapper_4/batch_normalization_4/batchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������R
 
_user_specified_nameinputs
�5
�
3__inference_module_wrapper_4_layer_call_fn_36603809

args_0K
=batch_normalization_4_assignmovingavg_readvariableop_resource:fM
?batch_normalization_4_assignmovingavg_1_readvariableop_resource:fI
;batch_normalization_4_batchnorm_mul_readvariableop_resource:fE
7batch_normalization_4_batchnorm_readvariableop_resource:f
identity��%batch_normalization_4/AssignMovingAvg�4batch_normalization_4/AssignMovingAvg/ReadVariableOp�'batch_normalization_4/AssignMovingAvg_1�6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp�.batch_normalization_4/batchnorm/ReadVariableOp�2batch_normalization_4/batchnorm/mul/ReadVariableOp~
4batch_normalization_4/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
"batch_normalization_4/moments/meanMeanargs_0=batch_normalization_4/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:f*
	keep_dims(�
*batch_normalization_4/moments/StopGradientStopGradient+batch_normalization_4/moments/mean:output:0*
T0*
_output_shapes

:f�
/batch_normalization_4/moments/SquaredDifferenceSquaredDifferenceargs_03batch_normalization_4/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������f�
8batch_normalization_4/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
&batch_normalization_4/moments/varianceMean3batch_normalization_4/moments/SquaredDifference:z:0Abatch_normalization_4/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:f*
	keep_dims(�
%batch_normalization_4/moments/SqueezeSqueeze+batch_normalization_4/moments/mean:output:0*
T0*
_output_shapes
:f*
squeeze_dims
 �
'batch_normalization_4/moments/Squeeze_1Squeeze/batch_normalization_4/moments/variance:output:0*
T0*
_output_shapes
:f*
squeeze_dims
 p
+batch_normalization_4/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
4batch_normalization_4/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_4_assignmovingavg_readvariableop_resource*
_output_shapes
:f*
dtype0�
)batch_normalization_4/AssignMovingAvg/subSub<batch_normalization_4/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_4/moments/Squeeze:output:0*
T0*
_output_shapes
:f�
)batch_normalization_4/AssignMovingAvg/mulMul-batch_normalization_4/AssignMovingAvg/sub:z:04batch_normalization_4/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:f�
%batch_normalization_4/AssignMovingAvgAssignSubVariableOp=batch_normalization_4_assignmovingavg_readvariableop_resource-batch_normalization_4/AssignMovingAvg/mul:z:05^batch_normalization_4/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0r
-batch_normalization_4/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_4_assignmovingavg_1_readvariableop_resource*
_output_shapes
:f*
dtype0�
+batch_normalization_4/AssignMovingAvg_1/subSub>batch_normalization_4/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_4/moments/Squeeze_1:output:0*
T0*
_output_shapes
:f�
+batch_normalization_4/AssignMovingAvg_1/mulMul/batch_normalization_4/AssignMovingAvg_1/sub:z:06batch_normalization_4/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:f�
'batch_normalization_4/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_4_assignmovingavg_1_readvariableop_resource/batch_normalization_4/AssignMovingAvg_1/mul:z:07^batch_normalization_4/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0j
%batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
#batch_normalization_4/batchnorm/addAddV20batch_normalization_4/moments/Squeeze_1:output:0.batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes
:f|
%batch_normalization_4/batchnorm/RsqrtRsqrt'batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes
:f�
2batch_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
:f*
dtype0�
#batch_normalization_4/batchnorm/mulMul)batch_normalization_4/batchnorm/Rsqrt:y:0:batch_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:f�
%batch_normalization_4/batchnorm/mul_1Mulargs_0'batch_normalization_4/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������f�
%batch_normalization_4/batchnorm/mul_2Mul.batch_normalization_4/moments/Squeeze:output:0'batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes
:f�
.batch_normalization_4/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
:f*
dtype0�
#batch_normalization_4/batchnorm/subSub6batch_normalization_4/batchnorm/ReadVariableOp:value:0)batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes
:f�
%batch_normalization_4/batchnorm/add_1AddV2)batch_normalization_4/batchnorm/mul_1:z:0'batch_normalization_4/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������fx
IdentityIdentity)batch_normalization_4/batchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������f�
NoOpNoOp&^batch_normalization_4/AssignMovingAvg5^batch_normalization_4/AssignMovingAvg/ReadVariableOp(^batch_normalization_4/AssignMovingAvg_17^batch_normalization_4/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_4/batchnorm/ReadVariableOp3^batch_normalization_4/batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������f: : : : 2N
%batch_normalization_4/AssignMovingAvg%batch_normalization_4/AssignMovingAvg2l
4batch_normalization_4/AssignMovingAvg/ReadVariableOp4batch_normalization_4/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_4/AssignMovingAvg_1'batch_normalization_4/AssignMovingAvg_12p
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_4/batchnorm/ReadVariableOp.batch_normalization_4/batchnorm/ReadVariableOp2h
2batch_normalization_4/batchnorm/mul/ReadVariableOp2batch_normalization_4/batchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������f
 
_user_specified_nameargs_0
�	
�
F__inference_dense_21_layer_call_and_return_conditional_losses_36603901

inputs1
matmul_readvariableop_resource:	f�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	f�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������f: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������f
 
_user_specified_nameinputs
�
�
N__inference_module_wrapper_4_layer_call_and_return_conditional_losses_36603829

args_0E
7batch_normalization_4_batchnorm_readvariableop_resource:fI
;batch_normalization_4_batchnorm_mul_readvariableop_resource:fG
9batch_normalization_4_batchnorm_readvariableop_1_resource:fG
9batch_normalization_4_batchnorm_readvariableop_2_resource:f
identity��.batch_normalization_4/batchnorm/ReadVariableOp�0batch_normalization_4/batchnorm/ReadVariableOp_1�0batch_normalization_4/batchnorm/ReadVariableOp_2�2batch_normalization_4/batchnorm/mul/ReadVariableOp�
.batch_normalization_4/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
:f*
dtype0j
%batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
#batch_normalization_4/batchnorm/addAddV26batch_normalization_4/batchnorm/ReadVariableOp:value:0.batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes
:f|
%batch_normalization_4/batchnorm/RsqrtRsqrt'batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes
:f�
2batch_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
:f*
dtype0�
#batch_normalization_4/batchnorm/mulMul)batch_normalization_4/batchnorm/Rsqrt:y:0:batch_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:f�
%batch_normalization_4/batchnorm/mul_1Mulargs_0'batch_normalization_4/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������f�
0batch_normalization_4/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_4_batchnorm_readvariableop_1_resource*
_output_shapes
:f*
dtype0�
%batch_normalization_4/batchnorm/mul_2Mul8batch_normalization_4/batchnorm/ReadVariableOp_1:value:0'batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes
:f�
0batch_normalization_4/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_4_batchnorm_readvariableop_2_resource*
_output_shapes
:f*
dtype0�
#batch_normalization_4/batchnorm/subSub8batch_normalization_4/batchnorm/ReadVariableOp_2:value:0)batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes
:f�
%batch_normalization_4/batchnorm/add_1AddV2)batch_normalization_4/batchnorm/mul_1:z:0'batch_normalization_4/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������fx
IdentityIdentity)batch_normalization_4/batchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������f�
NoOpNoOp/^batch_normalization_4/batchnorm/ReadVariableOp1^batch_normalization_4/batchnorm/ReadVariableOp_11^batch_normalization_4/batchnorm/ReadVariableOp_23^batch_normalization_4/batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������f: : : : 2`
.batch_normalization_4/batchnorm/ReadVariableOp.batch_normalization_4/batchnorm/ReadVariableOp2d
0batch_normalization_4/batchnorm/ReadVariableOp_10batch_normalization_4/batchnorm/ReadVariableOp_12d
0batch_normalization_4/batchnorm/ReadVariableOp_20batch_normalization_4/batchnorm/ReadVariableOp_22h
2batch_normalization_4/batchnorm/mul/ReadVariableOp2batch_normalization_4/batchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������f
 
_user_specified_nameargs_0
�
�
3__inference_module_wrapper_4_layer_call_fn_36603775

args_0E
7batch_normalization_4_batchnorm_readvariableop_resource:fI
;batch_normalization_4_batchnorm_mul_readvariableop_resource:fG
9batch_normalization_4_batchnorm_readvariableop_1_resource:fG
9batch_normalization_4_batchnorm_readvariableop_2_resource:f
identity��.batch_normalization_4/batchnorm/ReadVariableOp�0batch_normalization_4/batchnorm/ReadVariableOp_1�0batch_normalization_4/batchnorm/ReadVariableOp_2�2batch_normalization_4/batchnorm/mul/ReadVariableOp�
.batch_normalization_4/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
:f*
dtype0j
%batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
#batch_normalization_4/batchnorm/addAddV26batch_normalization_4/batchnorm/ReadVariableOp:value:0.batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes
:f|
%batch_normalization_4/batchnorm/RsqrtRsqrt'batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes
:f�
2batch_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
:f*
dtype0�
#batch_normalization_4/batchnorm/mulMul)batch_normalization_4/batchnorm/Rsqrt:y:0:batch_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:f�
%batch_normalization_4/batchnorm/mul_1Mulargs_0'batch_normalization_4/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������f�
0batch_normalization_4/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_4_batchnorm_readvariableop_1_resource*
_output_shapes
:f*
dtype0�
%batch_normalization_4/batchnorm/mul_2Mul8batch_normalization_4/batchnorm/ReadVariableOp_1:value:0'batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes
:f�
0batch_normalization_4/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_4_batchnorm_readvariableop_2_resource*
_output_shapes
:f*
dtype0�
#batch_normalization_4/batchnorm/subSub8batch_normalization_4/batchnorm/ReadVariableOp_2:value:0)batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes
:f�
%batch_normalization_4/batchnorm/add_1AddV2)batch_normalization_4/batchnorm/mul_1:z:0'batch_normalization_4/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������fx
IdentityIdentity)batch_normalization_4/batchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������f�
NoOpNoOp/^batch_normalization_4/batchnorm/ReadVariableOp1^batch_normalization_4/batchnorm/ReadVariableOp_11^batch_normalization_4/batchnorm/ReadVariableOp_23^batch_normalization_4/batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������f: : : : 2`
.batch_normalization_4/batchnorm/ReadVariableOp.batch_normalization_4/batchnorm/ReadVariableOp2d
0batch_normalization_4/batchnorm/ReadVariableOp_10batch_normalization_4/batchnorm/ReadVariableOp_12d
0batch_normalization_4/batchnorm/ReadVariableOp_20batch_normalization_4/batchnorm/ReadVariableOp_22h
2batch_normalization_4/batchnorm/mul/ReadVariableOp2batch_normalization_4/batchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������f
 
_user_specified_nameargs_0
�
c
G__inference_dropout_4_layer_call_and_return_conditional_losses_36603881

inputs
identityN
IdentityIdentityinputs*
T0*'
_output_shapes
:���������f"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������f:O K
'
_output_shapes
:���������f
 
_user_specified_nameinputs
�$
�
8__inference_batch_normalization_4_layer_call_fn_36604065

inputs5
'assignmovingavg_readvariableop_resource:f7
)assignmovingavg_1_readvariableop_resource:f3
%batchnorm_mul_readvariableop_resource:f/
!batchnorm_readvariableop_resource:f
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:f*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:f�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������fl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:f*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:f*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:f*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:f*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:fx
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:f�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:f*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:f~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:f�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:fP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:f~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:f*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:fc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������fh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:fv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:f*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:fr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������fb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������f�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������f: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������f
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_36604085

inputs/
!batchnorm_readvariableop_resource:f3
%batchnorm_mul_readvariableop_resource:f1
#batchnorm_readvariableop_1_resource:f1
#batchnorm_readvariableop_2_resource:f
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:f*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:fP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:f~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:f*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:fc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������fz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:f*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:fz
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:f*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:fr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������fb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������f�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������f: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������f
 
_user_specified_nameinputs
�

�
F__inference_dense_20_layer_call_and_return_conditional_losses_36603755

inputs0
matmul_readvariableop_resource:ff-
biasadd_readvariableop_resource:f
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:ff*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������fr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:f*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������fP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������fa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������fw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������f: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������f
 
_user_specified_nameinputs
�Q
�
#__inference__wrapped_model_36602772
dense_18_inputF
4sequential_4_dense_18_matmul_readvariableop_resource:RfC
5sequential_4_dense_18_biasadd_readvariableop_resource:fF
4sequential_4_dense_19_matmul_readvariableop_resource:ffC
5sequential_4_dense_19_biasadd_readvariableop_resource:fF
4sequential_4_dense_20_matmul_readvariableop_resource:ffC
5sequential_4_dense_20_biasadd_readvariableop_resource:fc
Usequential_4_module_wrapper_4_batch_normalization_4_batchnorm_readvariableop_resource:fg
Ysequential_4_module_wrapper_4_batch_normalization_4_batchnorm_mul_readvariableop_resource:fe
Wsequential_4_module_wrapper_4_batch_normalization_4_batchnorm_readvariableop_1_resource:fe
Wsequential_4_module_wrapper_4_batch_normalization_4_batchnorm_readvariableop_2_resource:fG
4sequential_4_dense_21_matmul_readvariableop_resource:	f�D
5sequential_4_dense_21_biasadd_readvariableop_resource:	�
identity��,sequential_4/dense_18/BiasAdd/ReadVariableOp�+sequential_4/dense_18/MatMul/ReadVariableOp�,sequential_4/dense_19/BiasAdd/ReadVariableOp�+sequential_4/dense_19/MatMul/ReadVariableOp�,sequential_4/dense_20/BiasAdd/ReadVariableOp�+sequential_4/dense_20/MatMul/ReadVariableOp�,sequential_4/dense_21/BiasAdd/ReadVariableOp�+sequential_4/dense_21/MatMul/ReadVariableOp�Lsequential_4/module_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp�Nsequential_4/module_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp_1�Nsequential_4/module_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp_2�Psequential_4/module_wrapper_4/batch_normalization_4/batchnorm/mul/ReadVariableOp�
+sequential_4/dense_18/MatMul/ReadVariableOpReadVariableOp4sequential_4_dense_18_matmul_readvariableop_resource*
_output_shapes

:Rf*
dtype0�
sequential_4/dense_18/MatMulMatMuldense_18_input3sequential_4/dense_18/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f�
,sequential_4/dense_18/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_dense_18_biasadd_readvariableop_resource*
_output_shapes
:f*
dtype0�
sequential_4/dense_18/BiasAddBiasAdd&sequential_4/dense_18/MatMul:product:04sequential_4/dense_18/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f|
sequential_4/dense_18/ReluRelu&sequential_4/dense_18/BiasAdd:output:0*
T0*'
_output_shapes
:���������f�
+sequential_4/dense_19/MatMul/ReadVariableOpReadVariableOp4sequential_4_dense_19_matmul_readvariableop_resource*
_output_shapes

:ff*
dtype0�
sequential_4/dense_19/MatMulMatMul(sequential_4/dense_18/Relu:activations:03sequential_4/dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f�
,sequential_4/dense_19/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_dense_19_biasadd_readvariableop_resource*
_output_shapes
:f*
dtype0�
sequential_4/dense_19/BiasAddBiasAdd&sequential_4/dense_19/MatMul:product:04sequential_4/dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f|
sequential_4/dense_19/ReluRelu&sequential_4/dense_19/BiasAdd:output:0*
T0*'
_output_shapes
:���������f�
+sequential_4/dense_20/MatMul/ReadVariableOpReadVariableOp4sequential_4_dense_20_matmul_readvariableop_resource*
_output_shapes

:ff*
dtype0�
sequential_4/dense_20/MatMulMatMul(sequential_4/dense_19/Relu:activations:03sequential_4/dense_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f�
,sequential_4/dense_20/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_dense_20_biasadd_readvariableop_resource*
_output_shapes
:f*
dtype0�
sequential_4/dense_20/BiasAddBiasAdd&sequential_4/dense_20/MatMul:product:04sequential_4/dense_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f|
sequential_4/dense_20/ReluRelu&sequential_4/dense_20/BiasAdd:output:0*
T0*'
_output_shapes
:���������f�
Lsequential_4/module_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOpReadVariableOpUsequential_4_module_wrapper_4_batch_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
:f*
dtype0�
Csequential_4/module_wrapper_4/batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
Asequential_4/module_wrapper_4/batch_normalization_4/batchnorm/addAddV2Tsequential_4/module_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp:value:0Lsequential_4/module_wrapper_4/batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes
:f�
Csequential_4/module_wrapper_4/batch_normalization_4/batchnorm/RsqrtRsqrtEsequential_4/module_wrapper_4/batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes
:f�
Psequential_4/module_wrapper_4/batch_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOpYsequential_4_module_wrapper_4_batch_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
:f*
dtype0�
Asequential_4/module_wrapper_4/batch_normalization_4/batchnorm/mulMulGsequential_4/module_wrapper_4/batch_normalization_4/batchnorm/Rsqrt:y:0Xsequential_4/module_wrapper_4/batch_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:f�
Csequential_4/module_wrapper_4/batch_normalization_4/batchnorm/mul_1Mul(sequential_4/dense_20/Relu:activations:0Esequential_4/module_wrapper_4/batch_normalization_4/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������f�
Nsequential_4/module_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp_1ReadVariableOpWsequential_4_module_wrapper_4_batch_normalization_4_batchnorm_readvariableop_1_resource*
_output_shapes
:f*
dtype0�
Csequential_4/module_wrapper_4/batch_normalization_4/batchnorm/mul_2MulVsequential_4/module_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp_1:value:0Esequential_4/module_wrapper_4/batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes
:f�
Nsequential_4/module_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp_2ReadVariableOpWsequential_4_module_wrapper_4_batch_normalization_4_batchnorm_readvariableop_2_resource*
_output_shapes
:f*
dtype0�
Asequential_4/module_wrapper_4/batch_normalization_4/batchnorm/subSubVsequential_4/module_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp_2:value:0Gsequential_4/module_wrapper_4/batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes
:f�
Csequential_4/module_wrapper_4/batch_normalization_4/batchnorm/add_1AddV2Gsequential_4/module_wrapper_4/batch_normalization_4/batchnorm/mul_1:z:0Esequential_4/module_wrapper_4/batch_normalization_4/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������f�
sequential_4/dropout_4/IdentityIdentityGsequential_4/module_wrapper_4/batch_normalization_4/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������f�
+sequential_4/dense_21/MatMul/ReadVariableOpReadVariableOp4sequential_4_dense_21_matmul_readvariableop_resource*
_output_shapes
:	f�*
dtype0�
sequential_4/dense_21/MatMulMatMul(sequential_4/dropout_4/Identity:output:03sequential_4/dense_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
,sequential_4/dense_21/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_dense_21_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential_4/dense_21/BiasAddBiasAdd&sequential_4/dense_21/MatMul:product:04sequential_4/dense_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������v
IdentityIdentity&sequential_4/dense_21/BiasAdd:output:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp-^sequential_4/dense_18/BiasAdd/ReadVariableOp,^sequential_4/dense_18/MatMul/ReadVariableOp-^sequential_4/dense_19/BiasAdd/ReadVariableOp,^sequential_4/dense_19/MatMul/ReadVariableOp-^sequential_4/dense_20/BiasAdd/ReadVariableOp,^sequential_4/dense_20/MatMul/ReadVariableOp-^sequential_4/dense_21/BiasAdd/ReadVariableOp,^sequential_4/dense_21/MatMul/ReadVariableOpM^sequential_4/module_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOpO^sequential_4/module_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp_1O^sequential_4/module_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp_2Q^sequential_4/module_wrapper_4/batch_normalization_4/batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������R: : : : : : : : : : : : 2\
,sequential_4/dense_18/BiasAdd/ReadVariableOp,sequential_4/dense_18/BiasAdd/ReadVariableOp2Z
+sequential_4/dense_18/MatMul/ReadVariableOp+sequential_4/dense_18/MatMul/ReadVariableOp2\
,sequential_4/dense_19/BiasAdd/ReadVariableOp,sequential_4/dense_19/BiasAdd/ReadVariableOp2Z
+sequential_4/dense_19/MatMul/ReadVariableOp+sequential_4/dense_19/MatMul/ReadVariableOp2\
,sequential_4/dense_20/BiasAdd/ReadVariableOp,sequential_4/dense_20/BiasAdd/ReadVariableOp2Z
+sequential_4/dense_20/MatMul/ReadVariableOp+sequential_4/dense_20/MatMul/ReadVariableOp2\
,sequential_4/dense_21/BiasAdd/ReadVariableOp,sequential_4/dense_21/BiasAdd/ReadVariableOp2Z
+sequential_4/dense_21/MatMul/ReadVariableOp+sequential_4/dense_21/MatMul/ReadVariableOp2�
Lsequential_4/module_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOpLsequential_4/module_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp2�
Nsequential_4/module_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp_1Nsequential_4/module_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp_12�
Nsequential_4/module_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp_2Nsequential_4/module_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp_22�
Psequential_4/module_wrapper_4/batch_normalization_4/batchnorm/mul/ReadVariableOpPsequential_4/module_wrapper_4/batch_normalization_4/batchnorm/mul/ReadVariableOp:W S
'
_output_shapes
:���������R
(
_user_specified_namedense_18_input
�

�
F__inference_dense_19_layer_call_and_return_conditional_losses_36603733

inputs0
matmul_readvariableop_resource:ff-
biasadd_readvariableop_resource:f
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:ff*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������fr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:f*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������fP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������fa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������fw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������f: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������f
 
_user_specified_nameinputs
�
J
,__inference_dropout_4_layer_call_fn_36603868

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:���������f[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������f"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������f:O K
'
_output_shapes
:���������f
 
_user_specified_nameinputs
�

�
+__inference_dense_20_layer_call_fn_36603744

inputs0
matmul_readvariableop_resource:ff-
biasadd_readvariableop_resource:f
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:ff*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������fr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:f*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������fP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������fa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������fw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������f: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������f
 
_user_specified_nameinputs
�
e
G__inference_dropout_4_layer_call_and_return_conditional_losses_36603877

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:���������f[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:���������f"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������f:O K
'
_output_shapes
:���������f
 
_user_specified_nameinputs
�
H
,__inference_dropout_4_layer_call_fn_36603872

inputs
identityN
IdentityIdentityinputs*
T0*'
_output_shapes
:���������f"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������f:O K
'
_output_shapes
:���������f
 
_user_specified_nameinputs
�C
�
J__inference_sequential_4_layer_call_and_return_conditional_losses_36603373
dense_18_input9
'dense_18_matmul_readvariableop_resource:Rf6
(dense_18_biasadd_readvariableop_resource:f9
'dense_19_matmul_readvariableop_resource:ff6
(dense_19_biasadd_readvariableop_resource:f9
'dense_20_matmul_readvariableop_resource:ff6
(dense_20_biasadd_readvariableop_resource:fV
Hmodule_wrapper_4_batch_normalization_4_batchnorm_readvariableop_resource:fZ
Lmodule_wrapper_4_batch_normalization_4_batchnorm_mul_readvariableop_resource:fX
Jmodule_wrapper_4_batch_normalization_4_batchnorm_readvariableop_1_resource:fX
Jmodule_wrapper_4_batch_normalization_4_batchnorm_readvariableop_2_resource:f:
'dense_21_matmul_readvariableop_resource:	f�7
(dense_21_biasadd_readvariableop_resource:	�
identity��dense_18/BiasAdd/ReadVariableOp�dense_18/MatMul/ReadVariableOp�dense_19/BiasAdd/ReadVariableOp�dense_19/MatMul/ReadVariableOp�dense_20/BiasAdd/ReadVariableOp�dense_20/MatMul/ReadVariableOp�dense_21/BiasAdd/ReadVariableOp�dense_21/MatMul/ReadVariableOp�?module_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp�Amodule_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp_1�Amodule_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp_2�Cmodule_wrapper_4/batch_normalization_4/batchnorm/mul/ReadVariableOp�
dense_18/MatMul/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource*
_output_shapes

:Rf*
dtype0�
dense_18/MatMulMatMuldense_18_input&dense_18/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f�
dense_18/BiasAdd/ReadVariableOpReadVariableOp(dense_18_biasadd_readvariableop_resource*
_output_shapes
:f*
dtype0�
dense_18/BiasAddBiasAdddense_18/MatMul:product:0'dense_18/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������fb
dense_18/ReluReludense_18/BiasAdd:output:0*
T0*'
_output_shapes
:���������f�
dense_19/MatMul/ReadVariableOpReadVariableOp'dense_19_matmul_readvariableop_resource*
_output_shapes

:ff*
dtype0�
dense_19/MatMulMatMuldense_18/Relu:activations:0&dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f�
dense_19/BiasAdd/ReadVariableOpReadVariableOp(dense_19_biasadd_readvariableop_resource*
_output_shapes
:f*
dtype0�
dense_19/BiasAddBiasAdddense_19/MatMul:product:0'dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������fb
dense_19/ReluReludense_19/BiasAdd:output:0*
T0*'
_output_shapes
:���������f�
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource*
_output_shapes

:ff*
dtype0�
dense_20/MatMulMatMuldense_19/Relu:activations:0&dense_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f�
dense_20/BiasAdd/ReadVariableOpReadVariableOp(dense_20_biasadd_readvariableop_resource*
_output_shapes
:f*
dtype0�
dense_20/BiasAddBiasAdddense_20/MatMul:product:0'dense_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������fb
dense_20/ReluReludense_20/BiasAdd:output:0*
T0*'
_output_shapes
:���������f�
?module_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOpReadVariableOpHmodule_wrapper_4_batch_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
:f*
dtype0{
6module_wrapper_4/batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
4module_wrapper_4/batch_normalization_4/batchnorm/addAddV2Gmodule_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp:value:0?module_wrapper_4/batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes
:f�
6module_wrapper_4/batch_normalization_4/batchnorm/RsqrtRsqrt8module_wrapper_4/batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes
:f�
Cmodule_wrapper_4/batch_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOpLmodule_wrapper_4_batch_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
:f*
dtype0�
4module_wrapper_4/batch_normalization_4/batchnorm/mulMul:module_wrapper_4/batch_normalization_4/batchnorm/Rsqrt:y:0Kmodule_wrapper_4/batch_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:f�
6module_wrapper_4/batch_normalization_4/batchnorm/mul_1Muldense_20/Relu:activations:08module_wrapper_4/batch_normalization_4/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������f�
Amodule_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp_1ReadVariableOpJmodule_wrapper_4_batch_normalization_4_batchnorm_readvariableop_1_resource*
_output_shapes
:f*
dtype0�
6module_wrapper_4/batch_normalization_4/batchnorm/mul_2MulImodule_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp_1:value:08module_wrapper_4/batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes
:f�
Amodule_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp_2ReadVariableOpJmodule_wrapper_4_batch_normalization_4_batchnorm_readvariableop_2_resource*
_output_shapes
:f*
dtype0�
4module_wrapper_4/batch_normalization_4/batchnorm/subSubImodule_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp_2:value:0:module_wrapper_4/batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes
:f�
6module_wrapper_4/batch_normalization_4/batchnorm/add_1AddV2:module_wrapper_4/batch_normalization_4/batchnorm/mul_1:z:08module_wrapper_4/batch_normalization_4/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������f�
dropout_4/IdentityIdentity:module_wrapper_4/batch_normalization_4/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������f�
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource*
_output_shapes
:	f�*
dtype0�
dense_21/MatMulMatMuldropout_4/Identity:output:0&dense_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������i
IdentityIdentitydense_21/BiasAdd:output:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp@^module_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOpB^module_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp_1B^module_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp_2D^module_wrapper_4/batch_normalization_4/batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������R: : : : : : : : : : : : 2B
dense_18/BiasAdd/ReadVariableOpdense_18/BiasAdd/ReadVariableOp2@
dense_18/MatMul/ReadVariableOpdense_18/MatMul/ReadVariableOp2B
dense_19/BiasAdd/ReadVariableOpdense_19/BiasAdd/ReadVariableOp2@
dense_19/MatMul/ReadVariableOpdense_19/MatMul/ReadVariableOp2B
dense_20/BiasAdd/ReadVariableOpdense_20/BiasAdd/ReadVariableOp2@
dense_20/MatMul/ReadVariableOpdense_20/MatMul/ReadVariableOp2B
dense_21/BiasAdd/ReadVariableOpdense_21/BiasAdd/ReadVariableOp2@
dense_21/MatMul/ReadVariableOpdense_21/MatMul/ReadVariableOp2�
?module_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp?module_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp2�
Amodule_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp_1Amodule_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp_12�
Amodule_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp_2Amodule_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp_22�
Cmodule_wrapper_4/batch_normalization_4/batchnorm/mul/ReadVariableOpCmodule_wrapper_4/batch_normalization_4/batchnorm/mul/ReadVariableOp:W S
'
_output_shapes
:���������R
(
_user_specified_namedense_18_input
�b
�
J__inference_sequential_4_layer_call_and_return_conditional_losses_36603658

inputs9
'dense_18_matmul_readvariableop_resource:Rf6
(dense_18_biasadd_readvariableop_resource:f9
'dense_19_matmul_readvariableop_resource:ff6
(dense_19_biasadd_readvariableop_resource:f9
'dense_20_matmul_readvariableop_resource:ff6
(dense_20_biasadd_readvariableop_resource:f\
Nmodule_wrapper_4_batch_normalization_4_assignmovingavg_readvariableop_resource:f^
Pmodule_wrapper_4_batch_normalization_4_assignmovingavg_1_readvariableop_resource:fZ
Lmodule_wrapper_4_batch_normalization_4_batchnorm_mul_readvariableop_resource:fV
Hmodule_wrapper_4_batch_normalization_4_batchnorm_readvariableop_resource:f:
'dense_21_matmul_readvariableop_resource:	f�7
(dense_21_biasadd_readvariableop_resource:	�
identity��dense_18/BiasAdd/ReadVariableOp�dense_18/MatMul/ReadVariableOp�dense_19/BiasAdd/ReadVariableOp�dense_19/MatMul/ReadVariableOp�dense_20/BiasAdd/ReadVariableOp�dense_20/MatMul/ReadVariableOp�dense_21/BiasAdd/ReadVariableOp�dense_21/MatMul/ReadVariableOp�6module_wrapper_4/batch_normalization_4/AssignMovingAvg�Emodule_wrapper_4/batch_normalization_4/AssignMovingAvg/ReadVariableOp�8module_wrapper_4/batch_normalization_4/AssignMovingAvg_1�Gmodule_wrapper_4/batch_normalization_4/AssignMovingAvg_1/ReadVariableOp�?module_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp�Cmodule_wrapper_4/batch_normalization_4/batchnorm/mul/ReadVariableOp�
dense_18/MatMul/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource*
_output_shapes

:Rf*
dtype0{
dense_18/MatMulMatMulinputs&dense_18/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f�
dense_18/BiasAdd/ReadVariableOpReadVariableOp(dense_18_biasadd_readvariableop_resource*
_output_shapes
:f*
dtype0�
dense_18/BiasAddBiasAdddense_18/MatMul:product:0'dense_18/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������fb
dense_18/ReluReludense_18/BiasAdd:output:0*
T0*'
_output_shapes
:���������f�
dense_19/MatMul/ReadVariableOpReadVariableOp'dense_19_matmul_readvariableop_resource*
_output_shapes

:ff*
dtype0�
dense_19/MatMulMatMuldense_18/Relu:activations:0&dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f�
dense_19/BiasAdd/ReadVariableOpReadVariableOp(dense_19_biasadd_readvariableop_resource*
_output_shapes
:f*
dtype0�
dense_19/BiasAddBiasAdddense_19/MatMul:product:0'dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������fb
dense_19/ReluReludense_19/BiasAdd:output:0*
T0*'
_output_shapes
:���������f�
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource*
_output_shapes

:ff*
dtype0�
dense_20/MatMulMatMuldense_19/Relu:activations:0&dense_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f�
dense_20/BiasAdd/ReadVariableOpReadVariableOp(dense_20_biasadd_readvariableop_resource*
_output_shapes
:f*
dtype0�
dense_20/BiasAddBiasAdddense_20/MatMul:product:0'dense_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������fb
dense_20/ReluReludense_20/BiasAdd:output:0*
T0*'
_output_shapes
:���������f�
Emodule_wrapper_4/batch_normalization_4/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
3module_wrapper_4/batch_normalization_4/moments/meanMeandense_20/Relu:activations:0Nmodule_wrapper_4/batch_normalization_4/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:f*
	keep_dims(�
;module_wrapper_4/batch_normalization_4/moments/StopGradientStopGradient<module_wrapper_4/batch_normalization_4/moments/mean:output:0*
T0*
_output_shapes

:f�
@module_wrapper_4/batch_normalization_4/moments/SquaredDifferenceSquaredDifferencedense_20/Relu:activations:0Dmodule_wrapper_4/batch_normalization_4/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������f�
Imodule_wrapper_4/batch_normalization_4/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
7module_wrapper_4/batch_normalization_4/moments/varianceMeanDmodule_wrapper_4/batch_normalization_4/moments/SquaredDifference:z:0Rmodule_wrapper_4/batch_normalization_4/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:f*
	keep_dims(�
6module_wrapper_4/batch_normalization_4/moments/SqueezeSqueeze<module_wrapper_4/batch_normalization_4/moments/mean:output:0*
T0*
_output_shapes
:f*
squeeze_dims
 �
8module_wrapper_4/batch_normalization_4/moments/Squeeze_1Squeeze@module_wrapper_4/batch_normalization_4/moments/variance:output:0*
T0*
_output_shapes
:f*
squeeze_dims
 �
<module_wrapper_4/batch_normalization_4/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
Emodule_wrapper_4/batch_normalization_4/AssignMovingAvg/ReadVariableOpReadVariableOpNmodule_wrapper_4_batch_normalization_4_assignmovingavg_readvariableop_resource*
_output_shapes
:f*
dtype0�
:module_wrapper_4/batch_normalization_4/AssignMovingAvg/subSubMmodule_wrapper_4/batch_normalization_4/AssignMovingAvg/ReadVariableOp:value:0?module_wrapper_4/batch_normalization_4/moments/Squeeze:output:0*
T0*
_output_shapes
:f�
:module_wrapper_4/batch_normalization_4/AssignMovingAvg/mulMul>module_wrapper_4/batch_normalization_4/AssignMovingAvg/sub:z:0Emodule_wrapper_4/batch_normalization_4/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:f�
6module_wrapper_4/batch_normalization_4/AssignMovingAvgAssignSubVariableOpNmodule_wrapper_4_batch_normalization_4_assignmovingavg_readvariableop_resource>module_wrapper_4/batch_normalization_4/AssignMovingAvg/mul:z:0F^module_wrapper_4/batch_normalization_4/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0�
>module_wrapper_4/batch_normalization_4/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
Gmodule_wrapper_4/batch_normalization_4/AssignMovingAvg_1/ReadVariableOpReadVariableOpPmodule_wrapper_4_batch_normalization_4_assignmovingavg_1_readvariableop_resource*
_output_shapes
:f*
dtype0�
<module_wrapper_4/batch_normalization_4/AssignMovingAvg_1/subSubOmodule_wrapper_4/batch_normalization_4/AssignMovingAvg_1/ReadVariableOp:value:0Amodule_wrapper_4/batch_normalization_4/moments/Squeeze_1:output:0*
T0*
_output_shapes
:f�
<module_wrapper_4/batch_normalization_4/AssignMovingAvg_1/mulMul@module_wrapper_4/batch_normalization_4/AssignMovingAvg_1/sub:z:0Gmodule_wrapper_4/batch_normalization_4/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:f�
8module_wrapper_4/batch_normalization_4/AssignMovingAvg_1AssignSubVariableOpPmodule_wrapper_4_batch_normalization_4_assignmovingavg_1_readvariableop_resource@module_wrapper_4/batch_normalization_4/AssignMovingAvg_1/mul:z:0H^module_wrapper_4/batch_normalization_4/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0{
6module_wrapper_4/batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
4module_wrapper_4/batch_normalization_4/batchnorm/addAddV2Amodule_wrapper_4/batch_normalization_4/moments/Squeeze_1:output:0?module_wrapper_4/batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes
:f�
6module_wrapper_4/batch_normalization_4/batchnorm/RsqrtRsqrt8module_wrapper_4/batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes
:f�
Cmodule_wrapper_4/batch_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOpLmodule_wrapper_4_batch_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
:f*
dtype0�
4module_wrapper_4/batch_normalization_4/batchnorm/mulMul:module_wrapper_4/batch_normalization_4/batchnorm/Rsqrt:y:0Kmodule_wrapper_4/batch_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:f�
6module_wrapper_4/batch_normalization_4/batchnorm/mul_1Muldense_20/Relu:activations:08module_wrapper_4/batch_normalization_4/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������f�
6module_wrapper_4/batch_normalization_4/batchnorm/mul_2Mul?module_wrapper_4/batch_normalization_4/moments/Squeeze:output:08module_wrapper_4/batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes
:f�
?module_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOpReadVariableOpHmodule_wrapper_4_batch_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
:f*
dtype0�
4module_wrapper_4/batch_normalization_4/batchnorm/subSubGmodule_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp:value:0:module_wrapper_4/batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes
:f�
6module_wrapper_4/batch_normalization_4/batchnorm/add_1AddV2:module_wrapper_4/batch_normalization_4/batchnorm/mul_1:z:08module_wrapper_4/batch_normalization_4/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������f�
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource*
_output_shapes
:	f�*
dtype0�
dense_21/MatMulMatMul:module_wrapper_4/batch_normalization_4/batchnorm/add_1:z:0&dense_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������i
IdentityIdentitydense_21/BiasAdd:output:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp7^module_wrapper_4/batch_normalization_4/AssignMovingAvgF^module_wrapper_4/batch_normalization_4/AssignMovingAvg/ReadVariableOp9^module_wrapper_4/batch_normalization_4/AssignMovingAvg_1H^module_wrapper_4/batch_normalization_4/AssignMovingAvg_1/ReadVariableOp@^module_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOpD^module_wrapper_4/batch_normalization_4/batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������R: : : : : : : : : : : : 2B
dense_18/BiasAdd/ReadVariableOpdense_18/BiasAdd/ReadVariableOp2@
dense_18/MatMul/ReadVariableOpdense_18/MatMul/ReadVariableOp2B
dense_19/BiasAdd/ReadVariableOpdense_19/BiasAdd/ReadVariableOp2@
dense_19/MatMul/ReadVariableOpdense_19/MatMul/ReadVariableOp2B
dense_20/BiasAdd/ReadVariableOpdense_20/BiasAdd/ReadVariableOp2@
dense_20/MatMul/ReadVariableOpdense_20/MatMul/ReadVariableOp2B
dense_21/BiasAdd/ReadVariableOpdense_21/BiasAdd/ReadVariableOp2@
dense_21/MatMul/ReadVariableOpdense_21/MatMul/ReadVariableOp2p
6module_wrapper_4/batch_normalization_4/AssignMovingAvg6module_wrapper_4/batch_normalization_4/AssignMovingAvg2�
Emodule_wrapper_4/batch_normalization_4/AssignMovingAvg/ReadVariableOpEmodule_wrapper_4/batch_normalization_4/AssignMovingAvg/ReadVariableOp2t
8module_wrapper_4/batch_normalization_4/AssignMovingAvg_18module_wrapper_4/batch_normalization_4/AssignMovingAvg_12�
Gmodule_wrapper_4/batch_normalization_4/AssignMovingAvg_1/ReadVariableOpGmodule_wrapper_4/batch_normalization_4/AssignMovingAvg_1/ReadVariableOp2�
?module_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp?module_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp2�
Cmodule_wrapper_4/batch_normalization_4/batchnorm/mul/ReadVariableOpCmodule_wrapper_4/batch_normalization_4/batchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������R
 
_user_specified_nameinputs
�

�
+__inference_dense_18_layer_call_fn_36603700

inputs0
matmul_readvariableop_resource:Rf-
biasadd_readvariableop_resource:f
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:Rf*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������fr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:f*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������fP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������fa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������fw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������R: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������R
 
_user_specified_nameinputs
�

�
+__inference_dense_19_layer_call_fn_36603722

inputs0
matmul_readvariableop_resource:ff-
biasadd_readvariableop_resource:f
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:ff*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������fr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:f*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������fP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������fa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������fw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������f: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������f
 
_user_specified_nameinputs
�b
�
/__inference_sequential_4_layer_call_fn_36603549

inputs9
'dense_18_matmul_readvariableop_resource:Rf6
(dense_18_biasadd_readvariableop_resource:f9
'dense_19_matmul_readvariableop_resource:ff6
(dense_19_biasadd_readvariableop_resource:f9
'dense_20_matmul_readvariableop_resource:ff6
(dense_20_biasadd_readvariableop_resource:f\
Nmodule_wrapper_4_batch_normalization_4_assignmovingavg_readvariableop_resource:f^
Pmodule_wrapper_4_batch_normalization_4_assignmovingavg_1_readvariableop_resource:fZ
Lmodule_wrapper_4_batch_normalization_4_batchnorm_mul_readvariableop_resource:fV
Hmodule_wrapper_4_batch_normalization_4_batchnorm_readvariableop_resource:f:
'dense_21_matmul_readvariableop_resource:	f�7
(dense_21_biasadd_readvariableop_resource:	�
identity��dense_18/BiasAdd/ReadVariableOp�dense_18/MatMul/ReadVariableOp�dense_19/BiasAdd/ReadVariableOp�dense_19/MatMul/ReadVariableOp�dense_20/BiasAdd/ReadVariableOp�dense_20/MatMul/ReadVariableOp�dense_21/BiasAdd/ReadVariableOp�dense_21/MatMul/ReadVariableOp�6module_wrapper_4/batch_normalization_4/AssignMovingAvg�Emodule_wrapper_4/batch_normalization_4/AssignMovingAvg/ReadVariableOp�8module_wrapper_4/batch_normalization_4/AssignMovingAvg_1�Gmodule_wrapper_4/batch_normalization_4/AssignMovingAvg_1/ReadVariableOp�?module_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp�Cmodule_wrapper_4/batch_normalization_4/batchnorm/mul/ReadVariableOp�
dense_18/MatMul/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource*
_output_shapes

:Rf*
dtype0{
dense_18/MatMulMatMulinputs&dense_18/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f�
dense_18/BiasAdd/ReadVariableOpReadVariableOp(dense_18_biasadd_readvariableop_resource*
_output_shapes
:f*
dtype0�
dense_18/BiasAddBiasAdddense_18/MatMul:product:0'dense_18/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������fb
dense_18/ReluReludense_18/BiasAdd:output:0*
T0*'
_output_shapes
:���������f�
dense_19/MatMul/ReadVariableOpReadVariableOp'dense_19_matmul_readvariableop_resource*
_output_shapes

:ff*
dtype0�
dense_19/MatMulMatMuldense_18/Relu:activations:0&dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f�
dense_19/BiasAdd/ReadVariableOpReadVariableOp(dense_19_biasadd_readvariableop_resource*
_output_shapes
:f*
dtype0�
dense_19/BiasAddBiasAdddense_19/MatMul:product:0'dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������fb
dense_19/ReluReludense_19/BiasAdd:output:0*
T0*'
_output_shapes
:���������f�
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource*
_output_shapes

:ff*
dtype0�
dense_20/MatMulMatMuldense_19/Relu:activations:0&dense_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f�
dense_20/BiasAdd/ReadVariableOpReadVariableOp(dense_20_biasadd_readvariableop_resource*
_output_shapes
:f*
dtype0�
dense_20/BiasAddBiasAdddense_20/MatMul:product:0'dense_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������fb
dense_20/ReluReludense_20/BiasAdd:output:0*
T0*'
_output_shapes
:���������f�
Emodule_wrapper_4/batch_normalization_4/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
3module_wrapper_4/batch_normalization_4/moments/meanMeandense_20/Relu:activations:0Nmodule_wrapper_4/batch_normalization_4/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:f*
	keep_dims(�
;module_wrapper_4/batch_normalization_4/moments/StopGradientStopGradient<module_wrapper_4/batch_normalization_4/moments/mean:output:0*
T0*
_output_shapes

:f�
@module_wrapper_4/batch_normalization_4/moments/SquaredDifferenceSquaredDifferencedense_20/Relu:activations:0Dmodule_wrapper_4/batch_normalization_4/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������f�
Imodule_wrapper_4/batch_normalization_4/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
7module_wrapper_4/batch_normalization_4/moments/varianceMeanDmodule_wrapper_4/batch_normalization_4/moments/SquaredDifference:z:0Rmodule_wrapper_4/batch_normalization_4/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:f*
	keep_dims(�
6module_wrapper_4/batch_normalization_4/moments/SqueezeSqueeze<module_wrapper_4/batch_normalization_4/moments/mean:output:0*
T0*
_output_shapes
:f*
squeeze_dims
 �
8module_wrapper_4/batch_normalization_4/moments/Squeeze_1Squeeze@module_wrapper_4/batch_normalization_4/moments/variance:output:0*
T0*
_output_shapes
:f*
squeeze_dims
 �
<module_wrapper_4/batch_normalization_4/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
Emodule_wrapper_4/batch_normalization_4/AssignMovingAvg/ReadVariableOpReadVariableOpNmodule_wrapper_4_batch_normalization_4_assignmovingavg_readvariableop_resource*
_output_shapes
:f*
dtype0�
:module_wrapper_4/batch_normalization_4/AssignMovingAvg/subSubMmodule_wrapper_4/batch_normalization_4/AssignMovingAvg/ReadVariableOp:value:0?module_wrapper_4/batch_normalization_4/moments/Squeeze:output:0*
T0*
_output_shapes
:f�
:module_wrapper_4/batch_normalization_4/AssignMovingAvg/mulMul>module_wrapper_4/batch_normalization_4/AssignMovingAvg/sub:z:0Emodule_wrapper_4/batch_normalization_4/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:f�
6module_wrapper_4/batch_normalization_4/AssignMovingAvgAssignSubVariableOpNmodule_wrapper_4_batch_normalization_4_assignmovingavg_readvariableop_resource>module_wrapper_4/batch_normalization_4/AssignMovingAvg/mul:z:0F^module_wrapper_4/batch_normalization_4/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0�
>module_wrapper_4/batch_normalization_4/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
Gmodule_wrapper_4/batch_normalization_4/AssignMovingAvg_1/ReadVariableOpReadVariableOpPmodule_wrapper_4_batch_normalization_4_assignmovingavg_1_readvariableop_resource*
_output_shapes
:f*
dtype0�
<module_wrapper_4/batch_normalization_4/AssignMovingAvg_1/subSubOmodule_wrapper_4/batch_normalization_4/AssignMovingAvg_1/ReadVariableOp:value:0Amodule_wrapper_4/batch_normalization_4/moments/Squeeze_1:output:0*
T0*
_output_shapes
:f�
<module_wrapper_4/batch_normalization_4/AssignMovingAvg_1/mulMul@module_wrapper_4/batch_normalization_4/AssignMovingAvg_1/sub:z:0Gmodule_wrapper_4/batch_normalization_4/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:f�
8module_wrapper_4/batch_normalization_4/AssignMovingAvg_1AssignSubVariableOpPmodule_wrapper_4_batch_normalization_4_assignmovingavg_1_readvariableop_resource@module_wrapper_4/batch_normalization_4/AssignMovingAvg_1/mul:z:0H^module_wrapper_4/batch_normalization_4/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0{
6module_wrapper_4/batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
4module_wrapper_4/batch_normalization_4/batchnorm/addAddV2Amodule_wrapper_4/batch_normalization_4/moments/Squeeze_1:output:0?module_wrapper_4/batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes
:f�
6module_wrapper_4/batch_normalization_4/batchnorm/RsqrtRsqrt8module_wrapper_4/batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes
:f�
Cmodule_wrapper_4/batch_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOpLmodule_wrapper_4_batch_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
:f*
dtype0�
4module_wrapper_4/batch_normalization_4/batchnorm/mulMul:module_wrapper_4/batch_normalization_4/batchnorm/Rsqrt:y:0Kmodule_wrapper_4/batch_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:f�
6module_wrapper_4/batch_normalization_4/batchnorm/mul_1Muldense_20/Relu:activations:08module_wrapper_4/batch_normalization_4/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������f�
6module_wrapper_4/batch_normalization_4/batchnorm/mul_2Mul?module_wrapper_4/batch_normalization_4/moments/Squeeze:output:08module_wrapper_4/batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes
:f�
?module_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOpReadVariableOpHmodule_wrapper_4_batch_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
:f*
dtype0�
4module_wrapper_4/batch_normalization_4/batchnorm/subSubGmodule_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp:value:0:module_wrapper_4/batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes
:f�
6module_wrapper_4/batch_normalization_4/batchnorm/add_1AddV2:module_wrapper_4/batch_normalization_4/batchnorm/mul_1:z:08module_wrapper_4/batch_normalization_4/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������f�
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource*
_output_shapes
:	f�*
dtype0�
dense_21/MatMulMatMul:module_wrapper_4/batch_normalization_4/batchnorm/add_1:z:0&dense_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������i
IdentityIdentitydense_21/BiasAdd:output:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp7^module_wrapper_4/batch_normalization_4/AssignMovingAvgF^module_wrapper_4/batch_normalization_4/AssignMovingAvg/ReadVariableOp9^module_wrapper_4/batch_normalization_4/AssignMovingAvg_1H^module_wrapper_4/batch_normalization_4/AssignMovingAvg_1/ReadVariableOp@^module_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOpD^module_wrapper_4/batch_normalization_4/batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������R: : : : : : : : : : : : 2B
dense_18/BiasAdd/ReadVariableOpdense_18/BiasAdd/ReadVariableOp2@
dense_18/MatMul/ReadVariableOpdense_18/MatMul/ReadVariableOp2B
dense_19/BiasAdd/ReadVariableOpdense_19/BiasAdd/ReadVariableOp2@
dense_19/MatMul/ReadVariableOpdense_19/MatMul/ReadVariableOp2B
dense_20/BiasAdd/ReadVariableOpdense_20/BiasAdd/ReadVariableOp2@
dense_20/MatMul/ReadVariableOpdense_20/MatMul/ReadVariableOp2B
dense_21/BiasAdd/ReadVariableOpdense_21/BiasAdd/ReadVariableOp2@
dense_21/MatMul/ReadVariableOpdense_21/MatMul/ReadVariableOp2p
6module_wrapper_4/batch_normalization_4/AssignMovingAvg6module_wrapper_4/batch_normalization_4/AssignMovingAvg2�
Emodule_wrapper_4/batch_normalization_4/AssignMovingAvg/ReadVariableOpEmodule_wrapper_4/batch_normalization_4/AssignMovingAvg/ReadVariableOp2t
8module_wrapper_4/batch_normalization_4/AssignMovingAvg_18module_wrapper_4/batch_normalization_4/AssignMovingAvg_12�
Gmodule_wrapper_4/batch_normalization_4/AssignMovingAvg_1/ReadVariableOpGmodule_wrapper_4/batch_normalization_4/AssignMovingAvg_1/ReadVariableOp2�
?module_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp?module_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp2�
Cmodule_wrapper_4/batch_normalization_4/batchnorm/mul/ReadVariableOpCmodule_wrapper_4/batch_normalization_4/batchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������R
 
_user_specified_nameinputs
�	
�
+__inference_dense_21_layer_call_fn_36603891

inputs1
matmul_readvariableop_resource:	f�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	f�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������f: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������f
 
_user_specified_nameinputs
�%
�
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_36604119

inputs5
'assignmovingavg_readvariableop_resource:f7
)assignmovingavg_1_readvariableop_resource:f3
%batchnorm_mul_readvariableop_resource:f/
!batchnorm_readvariableop_resource:f
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:f*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:f�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������fl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:f*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:f*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:f*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:f*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:fx
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:f�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:f*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:f~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:f�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:fP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:f~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:f*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:fc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������fh
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:fv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:f*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:fr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������fb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������f�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������f: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������f
 
_user_specified_nameinputs
�b
�
/__inference_sequential_4_layer_call_fn_36603325
dense_18_input9
'dense_18_matmul_readvariableop_resource:Rf6
(dense_18_biasadd_readvariableop_resource:f9
'dense_19_matmul_readvariableop_resource:ff6
(dense_19_biasadd_readvariableop_resource:f9
'dense_20_matmul_readvariableop_resource:ff6
(dense_20_biasadd_readvariableop_resource:f\
Nmodule_wrapper_4_batch_normalization_4_assignmovingavg_readvariableop_resource:f^
Pmodule_wrapper_4_batch_normalization_4_assignmovingavg_1_readvariableop_resource:fZ
Lmodule_wrapper_4_batch_normalization_4_batchnorm_mul_readvariableop_resource:fV
Hmodule_wrapper_4_batch_normalization_4_batchnorm_readvariableop_resource:f:
'dense_21_matmul_readvariableop_resource:	f�7
(dense_21_biasadd_readvariableop_resource:	�
identity��dense_18/BiasAdd/ReadVariableOp�dense_18/MatMul/ReadVariableOp�dense_19/BiasAdd/ReadVariableOp�dense_19/MatMul/ReadVariableOp�dense_20/BiasAdd/ReadVariableOp�dense_20/MatMul/ReadVariableOp�dense_21/BiasAdd/ReadVariableOp�dense_21/MatMul/ReadVariableOp�6module_wrapper_4/batch_normalization_4/AssignMovingAvg�Emodule_wrapper_4/batch_normalization_4/AssignMovingAvg/ReadVariableOp�8module_wrapper_4/batch_normalization_4/AssignMovingAvg_1�Gmodule_wrapper_4/batch_normalization_4/AssignMovingAvg_1/ReadVariableOp�?module_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp�Cmodule_wrapper_4/batch_normalization_4/batchnorm/mul/ReadVariableOp�
dense_18/MatMul/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource*
_output_shapes

:Rf*
dtype0�
dense_18/MatMulMatMuldense_18_input&dense_18/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f�
dense_18/BiasAdd/ReadVariableOpReadVariableOp(dense_18_biasadd_readvariableop_resource*
_output_shapes
:f*
dtype0�
dense_18/BiasAddBiasAdddense_18/MatMul:product:0'dense_18/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������fb
dense_18/ReluReludense_18/BiasAdd:output:0*
T0*'
_output_shapes
:���������f�
dense_19/MatMul/ReadVariableOpReadVariableOp'dense_19_matmul_readvariableop_resource*
_output_shapes

:ff*
dtype0�
dense_19/MatMulMatMuldense_18/Relu:activations:0&dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f�
dense_19/BiasAdd/ReadVariableOpReadVariableOp(dense_19_biasadd_readvariableop_resource*
_output_shapes
:f*
dtype0�
dense_19/BiasAddBiasAdddense_19/MatMul:product:0'dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������fb
dense_19/ReluReludense_19/BiasAdd:output:0*
T0*'
_output_shapes
:���������f�
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource*
_output_shapes

:ff*
dtype0�
dense_20/MatMulMatMuldense_19/Relu:activations:0&dense_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f�
dense_20/BiasAdd/ReadVariableOpReadVariableOp(dense_20_biasadd_readvariableop_resource*
_output_shapes
:f*
dtype0�
dense_20/BiasAddBiasAdddense_20/MatMul:product:0'dense_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������fb
dense_20/ReluReludense_20/BiasAdd:output:0*
T0*'
_output_shapes
:���������f�
Emodule_wrapper_4/batch_normalization_4/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
3module_wrapper_4/batch_normalization_4/moments/meanMeandense_20/Relu:activations:0Nmodule_wrapper_4/batch_normalization_4/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:f*
	keep_dims(�
;module_wrapper_4/batch_normalization_4/moments/StopGradientStopGradient<module_wrapper_4/batch_normalization_4/moments/mean:output:0*
T0*
_output_shapes

:f�
@module_wrapper_4/batch_normalization_4/moments/SquaredDifferenceSquaredDifferencedense_20/Relu:activations:0Dmodule_wrapper_4/batch_normalization_4/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������f�
Imodule_wrapper_4/batch_normalization_4/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
7module_wrapper_4/batch_normalization_4/moments/varianceMeanDmodule_wrapper_4/batch_normalization_4/moments/SquaredDifference:z:0Rmodule_wrapper_4/batch_normalization_4/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:f*
	keep_dims(�
6module_wrapper_4/batch_normalization_4/moments/SqueezeSqueeze<module_wrapper_4/batch_normalization_4/moments/mean:output:0*
T0*
_output_shapes
:f*
squeeze_dims
 �
8module_wrapper_4/batch_normalization_4/moments/Squeeze_1Squeeze@module_wrapper_4/batch_normalization_4/moments/variance:output:0*
T0*
_output_shapes
:f*
squeeze_dims
 �
<module_wrapper_4/batch_normalization_4/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
Emodule_wrapper_4/batch_normalization_4/AssignMovingAvg/ReadVariableOpReadVariableOpNmodule_wrapper_4_batch_normalization_4_assignmovingavg_readvariableop_resource*
_output_shapes
:f*
dtype0�
:module_wrapper_4/batch_normalization_4/AssignMovingAvg/subSubMmodule_wrapper_4/batch_normalization_4/AssignMovingAvg/ReadVariableOp:value:0?module_wrapper_4/batch_normalization_4/moments/Squeeze:output:0*
T0*
_output_shapes
:f�
:module_wrapper_4/batch_normalization_4/AssignMovingAvg/mulMul>module_wrapper_4/batch_normalization_4/AssignMovingAvg/sub:z:0Emodule_wrapper_4/batch_normalization_4/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:f�
6module_wrapper_4/batch_normalization_4/AssignMovingAvgAssignSubVariableOpNmodule_wrapper_4_batch_normalization_4_assignmovingavg_readvariableop_resource>module_wrapper_4/batch_normalization_4/AssignMovingAvg/mul:z:0F^module_wrapper_4/batch_normalization_4/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0�
>module_wrapper_4/batch_normalization_4/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
Gmodule_wrapper_4/batch_normalization_4/AssignMovingAvg_1/ReadVariableOpReadVariableOpPmodule_wrapper_4_batch_normalization_4_assignmovingavg_1_readvariableop_resource*
_output_shapes
:f*
dtype0�
<module_wrapper_4/batch_normalization_4/AssignMovingAvg_1/subSubOmodule_wrapper_4/batch_normalization_4/AssignMovingAvg_1/ReadVariableOp:value:0Amodule_wrapper_4/batch_normalization_4/moments/Squeeze_1:output:0*
T0*
_output_shapes
:f�
<module_wrapper_4/batch_normalization_4/AssignMovingAvg_1/mulMul@module_wrapper_4/batch_normalization_4/AssignMovingAvg_1/sub:z:0Gmodule_wrapper_4/batch_normalization_4/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:f�
8module_wrapper_4/batch_normalization_4/AssignMovingAvg_1AssignSubVariableOpPmodule_wrapper_4_batch_normalization_4_assignmovingavg_1_readvariableop_resource@module_wrapper_4/batch_normalization_4/AssignMovingAvg_1/mul:z:0H^module_wrapper_4/batch_normalization_4/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0{
6module_wrapper_4/batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
4module_wrapper_4/batch_normalization_4/batchnorm/addAddV2Amodule_wrapper_4/batch_normalization_4/moments/Squeeze_1:output:0?module_wrapper_4/batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes
:f�
6module_wrapper_4/batch_normalization_4/batchnorm/RsqrtRsqrt8module_wrapper_4/batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes
:f�
Cmodule_wrapper_4/batch_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOpLmodule_wrapper_4_batch_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
:f*
dtype0�
4module_wrapper_4/batch_normalization_4/batchnorm/mulMul:module_wrapper_4/batch_normalization_4/batchnorm/Rsqrt:y:0Kmodule_wrapper_4/batch_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:f�
6module_wrapper_4/batch_normalization_4/batchnorm/mul_1Muldense_20/Relu:activations:08module_wrapper_4/batch_normalization_4/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������f�
6module_wrapper_4/batch_normalization_4/batchnorm/mul_2Mul?module_wrapper_4/batch_normalization_4/moments/Squeeze:output:08module_wrapper_4/batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes
:f�
?module_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOpReadVariableOpHmodule_wrapper_4_batch_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
:f*
dtype0�
4module_wrapper_4/batch_normalization_4/batchnorm/subSubGmodule_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp:value:0:module_wrapper_4/batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes
:f�
6module_wrapper_4/batch_normalization_4/batchnorm/add_1AddV2:module_wrapper_4/batch_normalization_4/batchnorm/mul_1:z:08module_wrapper_4/batch_normalization_4/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������f�
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource*
_output_shapes
:	f�*
dtype0�
dense_21/MatMulMatMul:module_wrapper_4/batch_normalization_4/batchnorm/add_1:z:0&dense_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������i
IdentityIdentitydense_21/BiasAdd:output:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp7^module_wrapper_4/batch_normalization_4/AssignMovingAvgF^module_wrapper_4/batch_normalization_4/AssignMovingAvg/ReadVariableOp9^module_wrapper_4/batch_normalization_4/AssignMovingAvg_1H^module_wrapper_4/batch_normalization_4/AssignMovingAvg_1/ReadVariableOp@^module_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOpD^module_wrapper_4/batch_normalization_4/batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������R: : : : : : : : : : : : 2B
dense_18/BiasAdd/ReadVariableOpdense_18/BiasAdd/ReadVariableOp2@
dense_18/MatMul/ReadVariableOpdense_18/MatMul/ReadVariableOp2B
dense_19/BiasAdd/ReadVariableOpdense_19/BiasAdd/ReadVariableOp2@
dense_19/MatMul/ReadVariableOpdense_19/MatMul/ReadVariableOp2B
dense_20/BiasAdd/ReadVariableOpdense_20/BiasAdd/ReadVariableOp2@
dense_20/MatMul/ReadVariableOpdense_20/MatMul/ReadVariableOp2B
dense_21/BiasAdd/ReadVariableOpdense_21/BiasAdd/ReadVariableOp2@
dense_21/MatMul/ReadVariableOpdense_21/MatMul/ReadVariableOp2p
6module_wrapper_4/batch_normalization_4/AssignMovingAvg6module_wrapper_4/batch_normalization_4/AssignMovingAvg2�
Emodule_wrapper_4/batch_normalization_4/AssignMovingAvg/ReadVariableOpEmodule_wrapper_4/batch_normalization_4/AssignMovingAvg/ReadVariableOp2t
8module_wrapper_4/batch_normalization_4/AssignMovingAvg_18module_wrapper_4/batch_normalization_4/AssignMovingAvg_12�
Gmodule_wrapper_4/batch_normalization_4/AssignMovingAvg_1/ReadVariableOpGmodule_wrapper_4/batch_normalization_4/AssignMovingAvg_1/ReadVariableOp2�
?module_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp?module_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp2�
Cmodule_wrapper_4/batch_normalization_4/batchnorm/mul/ReadVariableOpCmodule_wrapper_4/batch_normalization_4/batchnorm/mul/ReadVariableOp:W S
'
_output_shapes
:���������R
(
_user_specified_namedense_18_input
�5
�
N__inference_module_wrapper_4_layer_call_and_return_conditional_losses_36603863

args_0K
=batch_normalization_4_assignmovingavg_readvariableop_resource:fM
?batch_normalization_4_assignmovingavg_1_readvariableop_resource:fI
;batch_normalization_4_batchnorm_mul_readvariableop_resource:fE
7batch_normalization_4_batchnorm_readvariableop_resource:f
identity��%batch_normalization_4/AssignMovingAvg�4batch_normalization_4/AssignMovingAvg/ReadVariableOp�'batch_normalization_4/AssignMovingAvg_1�6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp�.batch_normalization_4/batchnorm/ReadVariableOp�2batch_normalization_4/batchnorm/mul/ReadVariableOp~
4batch_normalization_4/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
"batch_normalization_4/moments/meanMeanargs_0=batch_normalization_4/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:f*
	keep_dims(�
*batch_normalization_4/moments/StopGradientStopGradient+batch_normalization_4/moments/mean:output:0*
T0*
_output_shapes

:f�
/batch_normalization_4/moments/SquaredDifferenceSquaredDifferenceargs_03batch_normalization_4/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������f�
8batch_normalization_4/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
&batch_normalization_4/moments/varianceMean3batch_normalization_4/moments/SquaredDifference:z:0Abatch_normalization_4/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:f*
	keep_dims(�
%batch_normalization_4/moments/SqueezeSqueeze+batch_normalization_4/moments/mean:output:0*
T0*
_output_shapes
:f*
squeeze_dims
 �
'batch_normalization_4/moments/Squeeze_1Squeeze/batch_normalization_4/moments/variance:output:0*
T0*
_output_shapes
:f*
squeeze_dims
 p
+batch_normalization_4/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
4batch_normalization_4/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_4_assignmovingavg_readvariableop_resource*
_output_shapes
:f*
dtype0�
)batch_normalization_4/AssignMovingAvg/subSub<batch_normalization_4/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_4/moments/Squeeze:output:0*
T0*
_output_shapes
:f�
)batch_normalization_4/AssignMovingAvg/mulMul-batch_normalization_4/AssignMovingAvg/sub:z:04batch_normalization_4/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:f�
%batch_normalization_4/AssignMovingAvgAssignSubVariableOp=batch_normalization_4_assignmovingavg_readvariableop_resource-batch_normalization_4/AssignMovingAvg/mul:z:05^batch_normalization_4/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0r
-batch_normalization_4/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_4_assignmovingavg_1_readvariableop_resource*
_output_shapes
:f*
dtype0�
+batch_normalization_4/AssignMovingAvg_1/subSub>batch_normalization_4/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_4/moments/Squeeze_1:output:0*
T0*
_output_shapes
:f�
+batch_normalization_4/AssignMovingAvg_1/mulMul/batch_normalization_4/AssignMovingAvg_1/sub:z:06batch_normalization_4/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:f�
'batch_normalization_4/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_4_assignmovingavg_1_readvariableop_resource/batch_normalization_4/AssignMovingAvg_1/mul:z:07^batch_normalization_4/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0j
%batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
#batch_normalization_4/batchnorm/addAddV20batch_normalization_4/moments/Squeeze_1:output:0.batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes
:f|
%batch_normalization_4/batchnorm/RsqrtRsqrt'batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes
:f�
2batch_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
:f*
dtype0�
#batch_normalization_4/batchnorm/mulMul)batch_normalization_4/batchnorm/Rsqrt:y:0:batch_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:f�
%batch_normalization_4/batchnorm/mul_1Mulargs_0'batch_normalization_4/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������f�
%batch_normalization_4/batchnorm/mul_2Mul.batch_normalization_4/moments/Squeeze:output:0'batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes
:f�
.batch_normalization_4/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
:f*
dtype0�
#batch_normalization_4/batchnorm/subSub6batch_normalization_4/batchnorm/ReadVariableOp:value:0)batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes
:f�
%batch_normalization_4/batchnorm/add_1AddV2)batch_normalization_4/batchnorm/mul_1:z:0'batch_normalization_4/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������fx
IdentityIdentity)batch_normalization_4/batchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������f�
NoOpNoOp&^batch_normalization_4/AssignMovingAvg5^batch_normalization_4/AssignMovingAvg/ReadVariableOp(^batch_normalization_4/AssignMovingAvg_17^batch_normalization_4/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_4/batchnorm/ReadVariableOp3^batch_normalization_4/batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������f: : : : 2N
%batch_normalization_4/AssignMovingAvg%batch_normalization_4/AssignMovingAvg2l
4batch_normalization_4/AssignMovingAvg/ReadVariableOp4batch_normalization_4/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_4/AssignMovingAvg_1'batch_normalization_4/AssignMovingAvg_12p
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_4/batchnorm/ReadVariableOp.batch_normalization_4/batchnorm/ReadVariableOp2h
2batch_normalization_4/batchnorm/mul/ReadVariableOp2batch_normalization_4/batchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������f
 
_user_specified_nameargs_0
�C
�
/__inference_sequential_4_layer_call_fn_36602821
dense_18_input9
'dense_18_matmul_readvariableop_resource:Rf6
(dense_18_biasadd_readvariableop_resource:f9
'dense_19_matmul_readvariableop_resource:ff6
(dense_19_biasadd_readvariableop_resource:f9
'dense_20_matmul_readvariableop_resource:ff6
(dense_20_biasadd_readvariableop_resource:fV
Hmodule_wrapper_4_batch_normalization_4_batchnorm_readvariableop_resource:fZ
Lmodule_wrapper_4_batch_normalization_4_batchnorm_mul_readvariableop_resource:fX
Jmodule_wrapper_4_batch_normalization_4_batchnorm_readvariableop_1_resource:fX
Jmodule_wrapper_4_batch_normalization_4_batchnorm_readvariableop_2_resource:f:
'dense_21_matmul_readvariableop_resource:	f�7
(dense_21_biasadd_readvariableop_resource:	�
identity��dense_18/BiasAdd/ReadVariableOp�dense_18/MatMul/ReadVariableOp�dense_19/BiasAdd/ReadVariableOp�dense_19/MatMul/ReadVariableOp�dense_20/BiasAdd/ReadVariableOp�dense_20/MatMul/ReadVariableOp�dense_21/BiasAdd/ReadVariableOp�dense_21/MatMul/ReadVariableOp�?module_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp�Amodule_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp_1�Amodule_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp_2�Cmodule_wrapper_4/batch_normalization_4/batchnorm/mul/ReadVariableOp�
dense_18/MatMul/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource*
_output_shapes

:Rf*
dtype0�
dense_18/MatMulMatMuldense_18_input&dense_18/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f�
dense_18/BiasAdd/ReadVariableOpReadVariableOp(dense_18_biasadd_readvariableop_resource*
_output_shapes
:f*
dtype0�
dense_18/BiasAddBiasAdddense_18/MatMul:product:0'dense_18/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������fb
dense_18/ReluReludense_18/BiasAdd:output:0*
T0*'
_output_shapes
:���������f�
dense_19/MatMul/ReadVariableOpReadVariableOp'dense_19_matmul_readvariableop_resource*
_output_shapes

:ff*
dtype0�
dense_19/MatMulMatMuldense_18/Relu:activations:0&dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f�
dense_19/BiasAdd/ReadVariableOpReadVariableOp(dense_19_biasadd_readvariableop_resource*
_output_shapes
:f*
dtype0�
dense_19/BiasAddBiasAdddense_19/MatMul:product:0'dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������fb
dense_19/ReluReludense_19/BiasAdd:output:0*
T0*'
_output_shapes
:���������f�
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource*
_output_shapes

:ff*
dtype0�
dense_20/MatMulMatMuldense_19/Relu:activations:0&dense_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f�
dense_20/BiasAdd/ReadVariableOpReadVariableOp(dense_20_biasadd_readvariableop_resource*
_output_shapes
:f*
dtype0�
dense_20/BiasAddBiasAdddense_20/MatMul:product:0'dense_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������fb
dense_20/ReluReludense_20/BiasAdd:output:0*
T0*'
_output_shapes
:���������f�
?module_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOpReadVariableOpHmodule_wrapper_4_batch_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
:f*
dtype0{
6module_wrapper_4/batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
4module_wrapper_4/batch_normalization_4/batchnorm/addAddV2Gmodule_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp:value:0?module_wrapper_4/batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes
:f�
6module_wrapper_4/batch_normalization_4/batchnorm/RsqrtRsqrt8module_wrapper_4/batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes
:f�
Cmodule_wrapper_4/batch_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOpLmodule_wrapper_4_batch_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
:f*
dtype0�
4module_wrapper_4/batch_normalization_4/batchnorm/mulMul:module_wrapper_4/batch_normalization_4/batchnorm/Rsqrt:y:0Kmodule_wrapper_4/batch_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:f�
6module_wrapper_4/batch_normalization_4/batchnorm/mul_1Muldense_20/Relu:activations:08module_wrapper_4/batch_normalization_4/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������f�
Amodule_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp_1ReadVariableOpJmodule_wrapper_4_batch_normalization_4_batchnorm_readvariableop_1_resource*
_output_shapes
:f*
dtype0�
6module_wrapper_4/batch_normalization_4/batchnorm/mul_2MulImodule_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp_1:value:08module_wrapper_4/batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes
:f�
Amodule_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp_2ReadVariableOpJmodule_wrapper_4_batch_normalization_4_batchnorm_readvariableop_2_resource*
_output_shapes
:f*
dtype0�
4module_wrapper_4/batch_normalization_4/batchnorm/subSubImodule_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp_2:value:0:module_wrapper_4/batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes
:f�
6module_wrapper_4/batch_normalization_4/batchnorm/add_1AddV2:module_wrapper_4/batch_normalization_4/batchnorm/mul_1:z:08module_wrapper_4/batch_normalization_4/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������f�
dropout_4/IdentityIdentity:module_wrapper_4/batch_normalization_4/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������f�
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource*
_output_shapes
:	f�*
dtype0�
dense_21/MatMulMatMuldropout_4/Identity:output:0&dense_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������i
IdentityIdentitydense_21/BiasAdd:output:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp@^module_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOpB^module_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp_1B^module_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp_2D^module_wrapper_4/batch_normalization_4/batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������R: : : : : : : : : : : : 2B
dense_18/BiasAdd/ReadVariableOpdense_18/BiasAdd/ReadVariableOp2@
dense_18/MatMul/ReadVariableOpdense_18/MatMul/ReadVariableOp2B
dense_19/BiasAdd/ReadVariableOpdense_19/BiasAdd/ReadVariableOp2@
dense_19/MatMul/ReadVariableOpdense_19/MatMul/ReadVariableOp2B
dense_20/BiasAdd/ReadVariableOpdense_20/BiasAdd/ReadVariableOp2@
dense_20/MatMul/ReadVariableOpdense_20/MatMul/ReadVariableOp2B
dense_21/BiasAdd/ReadVariableOpdense_21/BiasAdd/ReadVariableOp2@
dense_21/MatMul/ReadVariableOpdense_21/MatMul/ReadVariableOp2�
?module_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp?module_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp2�
Amodule_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp_1Amodule_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp_12�
Amodule_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp_2Amodule_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp_22�
Cmodule_wrapper_4/batch_normalization_4/batchnorm/mul/ReadVariableOpCmodule_wrapper_4/batch_normalization_4/batchnorm/mul/ReadVariableOp:W S
'
_output_shapes
:���������R
(
_user_specified_namedense_18_input
�

�
F__inference_dense_18_layer_call_and_return_conditional_losses_36603711

inputs0
matmul_readvariableop_resource:Rf-
biasadd_readvariableop_resource:f
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:Rf*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������fr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:f*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������fP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:���������fa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:���������fw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������R: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������R
 
_user_specified_nameinputs
�b
�
J__inference_sequential_4_layer_call_and_return_conditional_losses_36603434
dense_18_input9
'dense_18_matmul_readvariableop_resource:Rf6
(dense_18_biasadd_readvariableop_resource:f9
'dense_19_matmul_readvariableop_resource:ff6
(dense_19_biasadd_readvariableop_resource:f9
'dense_20_matmul_readvariableop_resource:ff6
(dense_20_biasadd_readvariableop_resource:f\
Nmodule_wrapper_4_batch_normalization_4_assignmovingavg_readvariableop_resource:f^
Pmodule_wrapper_4_batch_normalization_4_assignmovingavg_1_readvariableop_resource:fZ
Lmodule_wrapper_4_batch_normalization_4_batchnorm_mul_readvariableop_resource:fV
Hmodule_wrapper_4_batch_normalization_4_batchnorm_readvariableop_resource:f:
'dense_21_matmul_readvariableop_resource:	f�7
(dense_21_biasadd_readvariableop_resource:	�
identity��dense_18/BiasAdd/ReadVariableOp�dense_18/MatMul/ReadVariableOp�dense_19/BiasAdd/ReadVariableOp�dense_19/MatMul/ReadVariableOp�dense_20/BiasAdd/ReadVariableOp�dense_20/MatMul/ReadVariableOp�dense_21/BiasAdd/ReadVariableOp�dense_21/MatMul/ReadVariableOp�6module_wrapper_4/batch_normalization_4/AssignMovingAvg�Emodule_wrapper_4/batch_normalization_4/AssignMovingAvg/ReadVariableOp�8module_wrapper_4/batch_normalization_4/AssignMovingAvg_1�Gmodule_wrapper_4/batch_normalization_4/AssignMovingAvg_1/ReadVariableOp�?module_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp�Cmodule_wrapper_4/batch_normalization_4/batchnorm/mul/ReadVariableOp�
dense_18/MatMul/ReadVariableOpReadVariableOp'dense_18_matmul_readvariableop_resource*
_output_shapes

:Rf*
dtype0�
dense_18/MatMulMatMuldense_18_input&dense_18/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f�
dense_18/BiasAdd/ReadVariableOpReadVariableOp(dense_18_biasadd_readvariableop_resource*
_output_shapes
:f*
dtype0�
dense_18/BiasAddBiasAdddense_18/MatMul:product:0'dense_18/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������fb
dense_18/ReluReludense_18/BiasAdd:output:0*
T0*'
_output_shapes
:���������f�
dense_19/MatMul/ReadVariableOpReadVariableOp'dense_19_matmul_readvariableop_resource*
_output_shapes

:ff*
dtype0�
dense_19/MatMulMatMuldense_18/Relu:activations:0&dense_19/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f�
dense_19/BiasAdd/ReadVariableOpReadVariableOp(dense_19_biasadd_readvariableop_resource*
_output_shapes
:f*
dtype0�
dense_19/BiasAddBiasAdddense_19/MatMul:product:0'dense_19/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������fb
dense_19/ReluReludense_19/BiasAdd:output:0*
T0*'
_output_shapes
:���������f�
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource*
_output_shapes

:ff*
dtype0�
dense_20/MatMulMatMuldense_19/Relu:activations:0&dense_20/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f�
dense_20/BiasAdd/ReadVariableOpReadVariableOp(dense_20_biasadd_readvariableop_resource*
_output_shapes
:f*
dtype0�
dense_20/BiasAddBiasAdddense_20/MatMul:product:0'dense_20/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������fb
dense_20/ReluReludense_20/BiasAdd:output:0*
T0*'
_output_shapes
:���������f�
Emodule_wrapper_4/batch_normalization_4/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
3module_wrapper_4/batch_normalization_4/moments/meanMeandense_20/Relu:activations:0Nmodule_wrapper_4/batch_normalization_4/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:f*
	keep_dims(�
;module_wrapper_4/batch_normalization_4/moments/StopGradientStopGradient<module_wrapper_4/batch_normalization_4/moments/mean:output:0*
T0*
_output_shapes

:f�
@module_wrapper_4/batch_normalization_4/moments/SquaredDifferenceSquaredDifferencedense_20/Relu:activations:0Dmodule_wrapper_4/batch_normalization_4/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������f�
Imodule_wrapper_4/batch_normalization_4/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
7module_wrapper_4/batch_normalization_4/moments/varianceMeanDmodule_wrapper_4/batch_normalization_4/moments/SquaredDifference:z:0Rmodule_wrapper_4/batch_normalization_4/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:f*
	keep_dims(�
6module_wrapper_4/batch_normalization_4/moments/SqueezeSqueeze<module_wrapper_4/batch_normalization_4/moments/mean:output:0*
T0*
_output_shapes
:f*
squeeze_dims
 �
8module_wrapper_4/batch_normalization_4/moments/Squeeze_1Squeeze@module_wrapper_4/batch_normalization_4/moments/variance:output:0*
T0*
_output_shapes
:f*
squeeze_dims
 �
<module_wrapper_4/batch_normalization_4/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
Emodule_wrapper_4/batch_normalization_4/AssignMovingAvg/ReadVariableOpReadVariableOpNmodule_wrapper_4_batch_normalization_4_assignmovingavg_readvariableop_resource*
_output_shapes
:f*
dtype0�
:module_wrapper_4/batch_normalization_4/AssignMovingAvg/subSubMmodule_wrapper_4/batch_normalization_4/AssignMovingAvg/ReadVariableOp:value:0?module_wrapper_4/batch_normalization_4/moments/Squeeze:output:0*
T0*
_output_shapes
:f�
:module_wrapper_4/batch_normalization_4/AssignMovingAvg/mulMul>module_wrapper_4/batch_normalization_4/AssignMovingAvg/sub:z:0Emodule_wrapper_4/batch_normalization_4/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:f�
6module_wrapper_4/batch_normalization_4/AssignMovingAvgAssignSubVariableOpNmodule_wrapper_4_batch_normalization_4_assignmovingavg_readvariableop_resource>module_wrapper_4/batch_normalization_4/AssignMovingAvg/mul:z:0F^module_wrapper_4/batch_normalization_4/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0�
>module_wrapper_4/batch_normalization_4/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
Gmodule_wrapper_4/batch_normalization_4/AssignMovingAvg_1/ReadVariableOpReadVariableOpPmodule_wrapper_4_batch_normalization_4_assignmovingavg_1_readvariableop_resource*
_output_shapes
:f*
dtype0�
<module_wrapper_4/batch_normalization_4/AssignMovingAvg_1/subSubOmodule_wrapper_4/batch_normalization_4/AssignMovingAvg_1/ReadVariableOp:value:0Amodule_wrapper_4/batch_normalization_4/moments/Squeeze_1:output:0*
T0*
_output_shapes
:f�
<module_wrapper_4/batch_normalization_4/AssignMovingAvg_1/mulMul@module_wrapper_4/batch_normalization_4/AssignMovingAvg_1/sub:z:0Gmodule_wrapper_4/batch_normalization_4/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:f�
8module_wrapper_4/batch_normalization_4/AssignMovingAvg_1AssignSubVariableOpPmodule_wrapper_4_batch_normalization_4_assignmovingavg_1_readvariableop_resource@module_wrapper_4/batch_normalization_4/AssignMovingAvg_1/mul:z:0H^module_wrapper_4/batch_normalization_4/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0{
6module_wrapper_4/batch_normalization_4/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
4module_wrapper_4/batch_normalization_4/batchnorm/addAddV2Amodule_wrapper_4/batch_normalization_4/moments/Squeeze_1:output:0?module_wrapper_4/batch_normalization_4/batchnorm/add/y:output:0*
T0*
_output_shapes
:f�
6module_wrapper_4/batch_normalization_4/batchnorm/RsqrtRsqrt8module_wrapper_4/batch_normalization_4/batchnorm/add:z:0*
T0*
_output_shapes
:f�
Cmodule_wrapper_4/batch_normalization_4/batchnorm/mul/ReadVariableOpReadVariableOpLmodule_wrapper_4_batch_normalization_4_batchnorm_mul_readvariableop_resource*
_output_shapes
:f*
dtype0�
4module_wrapper_4/batch_normalization_4/batchnorm/mulMul:module_wrapper_4/batch_normalization_4/batchnorm/Rsqrt:y:0Kmodule_wrapper_4/batch_normalization_4/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:f�
6module_wrapper_4/batch_normalization_4/batchnorm/mul_1Muldense_20/Relu:activations:08module_wrapper_4/batch_normalization_4/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������f�
6module_wrapper_4/batch_normalization_4/batchnorm/mul_2Mul?module_wrapper_4/batch_normalization_4/moments/Squeeze:output:08module_wrapper_4/batch_normalization_4/batchnorm/mul:z:0*
T0*
_output_shapes
:f�
?module_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOpReadVariableOpHmodule_wrapper_4_batch_normalization_4_batchnorm_readvariableop_resource*
_output_shapes
:f*
dtype0�
4module_wrapper_4/batch_normalization_4/batchnorm/subSubGmodule_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp:value:0:module_wrapper_4/batch_normalization_4/batchnorm/mul_2:z:0*
T0*
_output_shapes
:f�
6module_wrapper_4/batch_normalization_4/batchnorm/add_1AddV2:module_wrapper_4/batch_normalization_4/batchnorm/mul_1:z:08module_wrapper_4/batch_normalization_4/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������f�
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource*
_output_shapes
:	f�*
dtype0�
dense_21/MatMulMatMul:module_wrapper_4/batch_normalization_4/batchnorm/add_1:z:0&dense_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������i
IdentityIdentitydense_21/BiasAdd:output:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp ^dense_18/BiasAdd/ReadVariableOp^dense_18/MatMul/ReadVariableOp ^dense_19/BiasAdd/ReadVariableOp^dense_19/MatMul/ReadVariableOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp7^module_wrapper_4/batch_normalization_4/AssignMovingAvgF^module_wrapper_4/batch_normalization_4/AssignMovingAvg/ReadVariableOp9^module_wrapper_4/batch_normalization_4/AssignMovingAvg_1H^module_wrapper_4/batch_normalization_4/AssignMovingAvg_1/ReadVariableOp@^module_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOpD^module_wrapper_4/batch_normalization_4/batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������R: : : : : : : : : : : : 2B
dense_18/BiasAdd/ReadVariableOpdense_18/BiasAdd/ReadVariableOp2@
dense_18/MatMul/ReadVariableOpdense_18/MatMul/ReadVariableOp2B
dense_19/BiasAdd/ReadVariableOpdense_19/BiasAdd/ReadVariableOp2@
dense_19/MatMul/ReadVariableOpdense_19/MatMul/ReadVariableOp2B
dense_20/BiasAdd/ReadVariableOpdense_20/BiasAdd/ReadVariableOp2@
dense_20/MatMul/ReadVariableOpdense_20/MatMul/ReadVariableOp2B
dense_21/BiasAdd/ReadVariableOpdense_21/BiasAdd/ReadVariableOp2@
dense_21/MatMul/ReadVariableOpdense_21/MatMul/ReadVariableOp2p
6module_wrapper_4/batch_normalization_4/AssignMovingAvg6module_wrapper_4/batch_normalization_4/AssignMovingAvg2�
Emodule_wrapper_4/batch_normalization_4/AssignMovingAvg/ReadVariableOpEmodule_wrapper_4/batch_normalization_4/AssignMovingAvg/ReadVariableOp2t
8module_wrapper_4/batch_normalization_4/AssignMovingAvg_18module_wrapper_4/batch_normalization_4/AssignMovingAvg_12�
Gmodule_wrapper_4/batch_normalization_4/AssignMovingAvg_1/ReadVariableOpGmodule_wrapper_4/batch_normalization_4/AssignMovingAvg_1/ReadVariableOp2�
?module_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp?module_wrapper_4/batch_normalization_4/batchnorm/ReadVariableOp2�
Cmodule_wrapper_4/batch_normalization_4/batchnorm/mul/ReadVariableOpCmodule_wrapper_4/batch_normalization_4/batchnorm/mul/ReadVariableOp:W S
'
_output_shapes
:���������R
(
_user_specified_namedense_18_input"�-
saver_filename:0
Identity:0Identity_448"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
I
dense_18_input7
 serving_default_dense_18_input:0���������R=
dense_211
StatefulPartitionedCall:0����������tensorflow/serving/predict:�
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer-4
layer_with_weights-4
layer-5
	optimizer
trainable_variables
	regularization_losses

	variables
	keras_api
_default_save_signature
__call__
*&call_and_return_all_conditional_losses

signatures"
_tf_keras_sequential
�

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
�

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
�

 kernel
!bias
"trainable_variables
#regularization_losses
$	variables
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses"
_tf_keras_layer
�
(_module
)trainable_variables
*regularization_losses
+	variables
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses"
_tf_keras_layer
�
/trainable_variables
0regularization_losses
1	variables
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses"
_tf_keras_layer
�

5kernel
6bias
7trainable_variables
8regularization_losses
9	variables
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses"
_tf_keras_layer
�
=iter

>beta_1

?beta_2
	@decay
Alearning_ratem�m�m�m� m�!m�5m�6m�Bm�Cm�v�v�v�v� v�!v�5v�6v�Bv�Cv�"
tf_deprecated_optimizer
f
0
1
2
3
 4
!5
B6
C7
58
69"
trackable_list_wrapper
 "
trackable_list_wrapper
v
0
1
2
3
 4
!5
B6
C7
D8
E9
510
611"
trackable_list_wrapper
�
Flayer_metrics
Gmetrics
trainable_variables

Hlayers
	regularization_losses

	variables
Ilayer_regularization_losses
Jnon_trainable_variables
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�2�
#__inference__wrapped_model_36602772�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *-�*
(�%
dense_18_input���������R
�2�
/__inference_sequential_4_layer_call_fn_36602821
/__inference_sequential_4_layer_call_fn_36603488
/__inference_sequential_4_layer_call_fn_36603549
/__inference_sequential_4_layer_call_fn_36603325�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
J__inference_sequential_4_layer_call_and_return_conditional_losses_36603597
J__inference_sequential_4_layer_call_and_return_conditional_losses_36603658
J__inference_sequential_4_layer_call_and_return_conditional_losses_36603373
J__inference_sequential_4_layer_call_and_return_conditional_losses_36603434�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
,
Kserving_default"
signature_map
!:Rf2dense_18/kernel
:f2dense_18/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
Llayer_metrics
Mmetrics
trainable_variables

Nlayers
regularization_losses
	variables
Olayer_regularization_losses
Pnon_trainable_variables
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�2�
+__inference_dense_18_layer_call_fn_36603700�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
F__inference_dense_18_layer_call_and_return_conditional_losses_36603711�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
!:ff2dense_19/kernel
:f2dense_19/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
Qlayer_metrics
Rmetrics
trainable_variables

Slayers
regularization_losses
	variables
Tlayer_regularization_losses
Unon_trainable_variables
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�2�
+__inference_dense_19_layer_call_fn_36603722�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
F__inference_dense_19_layer_call_and_return_conditional_losses_36603733�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
!:ff2dense_20/kernel
:f2dense_20/bias
.
 0
!1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
�
Vlayer_metrics
Wmetrics
"trainable_variables

Xlayers
#regularization_losses
$	variables
Ylayer_regularization_losses
Znon_trainable_variables
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses"
_generic_user_object
�2�
+__inference_dense_20_layer_call_fn_36603744�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
F__inference_dense_20_layer_call_and_return_conditional_losses_36603755�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
[axis
	Bgamma
Cbeta
Dmoving_mean
Emoving_variance
\	variables
]trainable_variables
^regularization_losses
_	keras_api
`__call__
*a&call_and_return_all_conditional_losses"
_tf_keras_layer
.
B0
C1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
B0
C1
D2
E3"
trackable_list_wrapper
�
blayer_metrics
cmetrics
)trainable_variables

dlayers
*regularization_losses
+	variables
elayer_regularization_losses
fnon_trainable_variables
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses"
_generic_user_object
�2�
3__inference_module_wrapper_4_layer_call_fn_36603775
3__inference_module_wrapper_4_layer_call_fn_36603809�
���
FullArgSpec
args�
jself
varargsjargs
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�2�
N__inference_module_wrapper_4_layer_call_and_return_conditional_losses_36603829
N__inference_module_wrapper_4_layer_call_and_return_conditional_losses_36603863�
���
FullArgSpec
args�
jself
varargsjargs
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
glayer_metrics
hmetrics
/trainable_variables

ilayers
0regularization_losses
1	variables
jlayer_regularization_losses
knon_trainable_variables
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses"
_generic_user_object
�2�
,__inference_dropout_4_layer_call_fn_36603868
,__inference_dropout_4_layer_call_fn_36603872�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
G__inference_dropout_4_layer_call_and_return_conditional_losses_36603877
G__inference_dropout_4_layer_call_and_return_conditional_losses_36603881�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
": 	f�2dense_21/kernel
:�2dense_21/bias
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
�
llayer_metrics
mmetrics
7trainable_variables

nlayers
8regularization_losses
9	variables
olayer_regularization_losses
pnon_trainable_variables
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses"
_generic_user_object
�2�
+__inference_dense_21_layer_call_fn_36603891�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
F__inference_dense_21_layer_call_and_return_conditional_losses_36603901�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
::8f2,module_wrapper_4/batch_normalization_4/gamma
9:7f2+module_wrapper_4/batch_normalization_4/beta
B:@f (22module_wrapper_4/batch_normalization_4/moving_mean
F:Df (26module_wrapper_4/batch_normalization_4/moving_variance
 "
trackable_dict_wrapper
.
q0
r1"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
�B�
&__inference_signature_wrapper_36603689dense_18_input"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
trackable_list_wrapper
<
B0
C1
D2
E3"
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
snon_trainable_variables

tlayers
umetrics
vlayer_regularization_losses
wlayer_metrics
\	variables
]trainable_variables
^regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses"
_generic_user_object
�2�
8__inference_batch_normalization_4_layer_call_fn_36604031
8__inference_batch_normalization_4_layer_call_fn_36604065�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_36604085
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_36604119�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
D0
E1"
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
N
	xtotal
	ycount
z	variables
{	keras_api"
_tf_keras_metric
i
	|total
	}count
~_fn

_fn_kwargs
�	variables
�	keras_api"
_tf_keras_metric
.
D0
E1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:  (2total
:  (2count
.
x0
y1"
trackable_list_wrapper
-
z	variables"
_generic_user_object
:  (2total
:  (2count
c

�total

�count
�
_fn_kwargs
�	variables
�	keras_api"
_tf_keras_metric
 "
trackable_dict_wrapper
.
|0
}1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
&:$Rf2Adam/dense_18/kernel/m
 :f2Adam/dense_18/bias/m
&:$ff2Adam/dense_19/kernel/m
 :f2Adam/dense_19/bias/m
&:$ff2Adam/dense_20/kernel/m
 :f2Adam/dense_20/bias/m
':%	f�2Adam/dense_21/kernel/m
!:�2Adam/dense_21/bias/m
?:=f23Adam/module_wrapper_4/batch_normalization_4/gamma/m
>:<f22Adam/module_wrapper_4/batch_normalization_4/beta/m
&:$Rf2Adam/dense_18/kernel/v
 :f2Adam/dense_18/bias/v
&:$ff2Adam/dense_19/kernel/v
 :f2Adam/dense_19/bias/v
&:$ff2Adam/dense_20/kernel/v
 :f2Adam/dense_20/bias/v
':%	f�2Adam/dense_21/kernel/v
!:�2Adam/dense_21/bias/v
?:=f23Adam/module_wrapper_4/batch_normalization_4/gamma/v
>:<f22Adam/module_wrapper_4/batch_normalization_4/beta/v�
#__inference__wrapped_model_36602772} !EBDC567�4
-�*
(�%
dense_18_input���������R
� "4�1
/
dense_21#� 
dense_21�����������
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_36604085bEBDC3�0
)�&
 �
inputs���������f
p 
� "%�"
�
0���������f
� �
S__inference_batch_normalization_4_layer_call_and_return_conditional_losses_36604119bDEBC3�0
)�&
 �
inputs���������f
p
� "%�"
�
0���������f
� �
8__inference_batch_normalization_4_layer_call_fn_36604031UEBDC3�0
)�&
 �
inputs���������f
p 
� "����������f�
8__inference_batch_normalization_4_layer_call_fn_36604065UDEBC3�0
)�&
 �
inputs���������f
p
� "����������f�
F__inference_dense_18_layer_call_and_return_conditional_losses_36603711\/�,
%�"
 �
inputs���������R
� "%�"
�
0���������f
� ~
+__inference_dense_18_layer_call_fn_36603700O/�,
%�"
 �
inputs���������R
� "����������f�
F__inference_dense_19_layer_call_and_return_conditional_losses_36603733\/�,
%�"
 �
inputs���������f
� "%�"
�
0���������f
� ~
+__inference_dense_19_layer_call_fn_36603722O/�,
%�"
 �
inputs���������f
� "����������f�
F__inference_dense_20_layer_call_and_return_conditional_losses_36603755\ !/�,
%�"
 �
inputs���������f
� "%�"
�
0���������f
� ~
+__inference_dense_20_layer_call_fn_36603744O !/�,
%�"
 �
inputs���������f
� "����������f�
F__inference_dense_21_layer_call_and_return_conditional_losses_36603901]56/�,
%�"
 �
inputs���������f
� "&�#
�
0����������
� 
+__inference_dense_21_layer_call_fn_36603891P56/�,
%�"
 �
inputs���������f
� "������������
G__inference_dropout_4_layer_call_and_return_conditional_losses_36603877\3�0
)�&
 �
inputs���������f
p 
� "%�"
�
0���������f
� �
G__inference_dropout_4_layer_call_and_return_conditional_losses_36603881\3�0
)�&
 �
inputs���������f
p
� "%�"
�
0���������f
� 
,__inference_dropout_4_layer_call_fn_36603868O3�0
)�&
 �
inputs���������f
p 
� "����������f
,__inference_dropout_4_layer_call_fn_36603872O3�0
)�&
 �
inputs���������f
p
� "����������f�
N__inference_module_wrapper_4_layer_call_and_return_conditional_losses_36603829nEBDC?�<
%�"
 �
args_0���������f
�

trainingp "%�"
�
0���������f
� �
N__inference_module_wrapper_4_layer_call_and_return_conditional_losses_36603863nDEBC?�<
%�"
 �
args_0���������f
�

trainingp"%�"
�
0���������f
� �
3__inference_module_wrapper_4_layer_call_fn_36603775aEBDC?�<
%�"
 �
args_0���������f
�

trainingp "����������f�
3__inference_module_wrapper_4_layer_call_fn_36603809aDEBC?�<
%�"
 �
args_0���������f
�

trainingp"����������f�
J__inference_sequential_4_layer_call_and_return_conditional_losses_36603373w !EBDC56?�<
5�2
(�%
dense_18_input���������R
p 

 
� "&�#
�
0����������
� �
J__inference_sequential_4_layer_call_and_return_conditional_losses_36603434w !DEBC56?�<
5�2
(�%
dense_18_input���������R
p

 
� "&�#
�
0����������
� �
J__inference_sequential_4_layer_call_and_return_conditional_losses_36603597o !EBDC567�4
-�*
 �
inputs���������R
p 

 
� "&�#
�
0����������
� �
J__inference_sequential_4_layer_call_and_return_conditional_losses_36603658o !DEBC567�4
-�*
 �
inputs���������R
p

 
� "&�#
�
0����������
� �
/__inference_sequential_4_layer_call_fn_36602821j !EBDC56?�<
5�2
(�%
dense_18_input���������R
p 

 
� "������������
/__inference_sequential_4_layer_call_fn_36603325j !DEBC56?�<
5�2
(�%
dense_18_input���������R
p

 
� "������������
/__inference_sequential_4_layer_call_fn_36603488b !EBDC567�4
-�*
 �
inputs���������R
p 

 
� "������������
/__inference_sequential_4_layer_call_fn_36603549b !DEBC567�4
-�*
 �
inputs���������R
p

 
� "������������
&__inference_signature_wrapper_36603689� !EBDC56I�F
� 
?�<
:
dense_18_input(�%
dense_18_input���������R"4�1
/
dense_21#� 
dense_21����������