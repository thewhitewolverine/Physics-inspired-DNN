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
x
dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Rf*
shared_namedense_8/kernel
q
"dense_8/kernel/Read/ReadVariableOpReadVariableOpdense_8/kernel*
_output_shapes

:Rf*
dtype0
p
dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:f*
shared_namedense_8/bias
i
 dense_8/bias/Read/ReadVariableOpReadVariableOpdense_8/bias*
_output_shapes
:f*
dtype0
x
dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:ff*
shared_namedense_9/kernel
q
"dense_9/kernel/Read/ReadVariableOpReadVariableOpdense_9/kernel*
_output_shapes

:ff*
dtype0
p
dense_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:f*
shared_namedense_9/bias
i
 dense_9/bias/Read/ReadVariableOpReadVariableOpdense_9/bias*
_output_shapes
:f*
dtype0
z
dense_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:ff* 
shared_namedense_10/kernel
s
#dense_10/kernel/Read/ReadVariableOpReadVariableOpdense_10/kernel*
_output_shapes

:ff*
dtype0
r
dense_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:f*
shared_namedense_10/bias
k
!dense_10/bias/Read/ReadVariableOpReadVariableOpdense_10/bias*
_output_shapes
:f*
dtype0
{
dense_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	f�* 
shared_namedense_11/kernel
t
#dense_11/kernel/Read/ReadVariableOpReadVariableOpdense_11/kernel*
_output_shapes
:	f�*
dtype0
s
dense_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namedense_11/bias
l
!dense_11/bias/Read/ReadVariableOpReadVariableOpdense_11/bias*
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
,module_wrapper_2/batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:f*=
shared_name.,module_wrapper_2/batch_normalization_2/gamma
�
@module_wrapper_2/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOp,module_wrapper_2/batch_normalization_2/gamma*
_output_shapes
:f*
dtype0
�
+module_wrapper_2/batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:f*<
shared_name-+module_wrapper_2/batch_normalization_2/beta
�
?module_wrapper_2/batch_normalization_2/beta/Read/ReadVariableOpReadVariableOp+module_wrapper_2/batch_normalization_2/beta*
_output_shapes
:f*
dtype0
�
2module_wrapper_2/batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:f*C
shared_name42module_wrapper_2/batch_normalization_2/moving_mean
�
Fmodule_wrapper_2/batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp2module_wrapper_2/batch_normalization_2/moving_mean*
_output_shapes
:f*
dtype0
�
6module_wrapper_2/batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:f*G
shared_name86module_wrapper_2/batch_normalization_2/moving_variance
�
Jmodule_wrapper_2/batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp6module_wrapper_2/batch_normalization_2/moving_variance*
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
Adam/dense_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Rf*&
shared_nameAdam/dense_8/kernel/m

)Adam/dense_8/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_8/kernel/m*
_output_shapes

:Rf*
dtype0
~
Adam/dense_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:f*$
shared_nameAdam/dense_8/bias/m
w
'Adam/dense_8/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_8/bias/m*
_output_shapes
:f*
dtype0
�
Adam/dense_9/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:ff*&
shared_nameAdam/dense_9/kernel/m

)Adam/dense_9/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_9/kernel/m*
_output_shapes

:ff*
dtype0
~
Adam/dense_9/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:f*$
shared_nameAdam/dense_9/bias/m
w
'Adam/dense_9/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_9/bias/m*
_output_shapes
:f*
dtype0
�
Adam/dense_10/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:ff*'
shared_nameAdam/dense_10/kernel/m
�
*Adam/dense_10/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_10/kernel/m*
_output_shapes

:ff*
dtype0
�
Adam/dense_10/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:f*%
shared_nameAdam/dense_10/bias/m
y
(Adam/dense_10/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_10/bias/m*
_output_shapes
:f*
dtype0
�
Adam/dense_11/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	f�*'
shared_nameAdam/dense_11/kernel/m
�
*Adam/dense_11/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_11/kernel/m*
_output_shapes
:	f�*
dtype0
�
Adam/dense_11/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/dense_11/bias/m
z
(Adam/dense_11/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_11/bias/m*
_output_shapes	
:�*
dtype0
�
3Adam/module_wrapper_2/batch_normalization_2/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:f*D
shared_name53Adam/module_wrapper_2/batch_normalization_2/gamma/m
�
GAdam/module_wrapper_2/batch_normalization_2/gamma/m/Read/ReadVariableOpReadVariableOp3Adam/module_wrapper_2/batch_normalization_2/gamma/m*
_output_shapes
:f*
dtype0
�
2Adam/module_wrapper_2/batch_normalization_2/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:f*C
shared_name42Adam/module_wrapper_2/batch_normalization_2/beta/m
�
FAdam/module_wrapper_2/batch_normalization_2/beta/m/Read/ReadVariableOpReadVariableOp2Adam/module_wrapper_2/batch_normalization_2/beta/m*
_output_shapes
:f*
dtype0
�
Adam/dense_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:Rf*&
shared_nameAdam/dense_8/kernel/v

)Adam/dense_8/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_8/kernel/v*
_output_shapes

:Rf*
dtype0
~
Adam/dense_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:f*$
shared_nameAdam/dense_8/bias/v
w
'Adam/dense_8/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_8/bias/v*
_output_shapes
:f*
dtype0
�
Adam/dense_9/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:ff*&
shared_nameAdam/dense_9/kernel/v

)Adam/dense_9/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_9/kernel/v*
_output_shapes

:ff*
dtype0
~
Adam/dense_9/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:f*$
shared_nameAdam/dense_9/bias/v
w
'Adam/dense_9/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_9/bias/v*
_output_shapes
:f*
dtype0
�
Adam/dense_10/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:ff*'
shared_nameAdam/dense_10/kernel/v
�
*Adam/dense_10/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_10/kernel/v*
_output_shapes

:ff*
dtype0
�
Adam/dense_10/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:f*%
shared_nameAdam/dense_10/bias/v
y
(Adam/dense_10/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_10/bias/v*
_output_shapes
:f*
dtype0
�
Adam/dense_11/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	f�*'
shared_nameAdam/dense_11/kernel/v
�
*Adam/dense_11/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_11/kernel/v*
_output_shapes
:	f�*
dtype0
�
Adam/dense_11/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameAdam/dense_11/bias/v
z
(Adam/dense_11/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_11/bias/v*
_output_shapes	
:�*
dtype0
�
3Adam/module_wrapper_2/batch_normalization_2/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:f*D
shared_name53Adam/module_wrapper_2/batch_normalization_2/gamma/v
�
GAdam/module_wrapper_2/batch_normalization_2/gamma/v/Read/ReadVariableOpReadVariableOp3Adam/module_wrapper_2/batch_normalization_2/gamma/v*
_output_shapes
:f*
dtype0
�
2Adam/module_wrapper_2/batch_normalization_2/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:f*C
shared_name42Adam/module_wrapper_2/batch_normalization_2/beta/v
�
FAdam/module_wrapper_2/batch_normalization_2/beta/v/Read/ReadVariableOpReadVariableOp2Adam/module_wrapper_2/batch_normalization_2/beta/v*
_output_shapes
:f*
dtype0

NoOpNoOp
�P
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�O
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
regularization_losses
	trainable_variables

	variables
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
�

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
__call__
*&call_and_return_all_conditional_losses*
�

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
__call__
*&call_and_return_all_conditional_losses*
�

 kernel
!bias
"regularization_losses
#trainable_variables
$	variables
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses*
�
(_module
)regularization_losses
*trainable_variables
+	variables
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses*
�
/regularization_losses
0trainable_variables
1	variables
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses* 
�

5kernel
6bias
7regularization_losses
8trainable_variables
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
* 
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
regularization_losses
	trainable_variables

Flayers
Gnon_trainable_variables
Hmetrics

	variables
Ilayer_regularization_losses
Jlayer_metrics
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

Kserving_default* 
^X
VARIABLE_VALUEdense_8/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_8/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1*

0
1*
�
regularization_losses

Llayers
trainable_variables
	variables
Mnon_trainable_variables
Nmetrics
Olayer_regularization_losses
Player_metrics
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
^X
VARIABLE_VALUEdense_9/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_9/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1*

0
1*
�
regularization_losses

Qlayers
trainable_variables
	variables
Rnon_trainable_variables
Smetrics
Tlayer_regularization_losses
Ulayer_metrics
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
_Y
VARIABLE_VALUEdense_10/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_10/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

 0
!1*

 0
!1*
�
"regularization_losses

Vlayers
#trainable_variables
$	variables
Wnon_trainable_variables
Xmetrics
Ylayer_regularization_losses
Zlayer_metrics
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
* 

B0
C1*
 
B0
C1
D2
E3*
�
)regularization_losses

blayers
*trainable_variables
+	variables
cnon_trainable_variables
dmetrics
elayer_regularization_losses
flayer_metrics
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses*
* 
* 
* 
* 
* 
�
/regularization_losses

glayers
0trainable_variables
1	variables
hnon_trainable_variables
imetrics
jlayer_regularization_losses
klayer_metrics
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses* 
* 
* 
_Y
VARIABLE_VALUEdense_11/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_11/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

50
61*

50
61*
�
7regularization_losses

llayers
8trainable_variables
9	variables
mnon_trainable_variables
nmetrics
olayer_regularization_losses
player_metrics
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
VARIABLE_VALUE,module_wrapper_2/batch_normalization_2/gamma0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE+module_wrapper_2/batch_normalization_2/beta0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE2module_wrapper_2/batch_normalization_2/moving_mean&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE6module_wrapper_2/batch_normalization_2/moving_variance&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
.
0
1
2
3
4
5*

D0
E1*

q0
r1*
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
�{
VARIABLE_VALUEAdam/dense_8/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_8/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�{
VARIABLE_VALUEAdam/dense_9/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_9/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/dense_10/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_10/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/dense_11/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_11/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE3Adam/module_wrapper_2/batch_normalization_2/gamma/mLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE2Adam/module_wrapper_2/batch_normalization_2/beta/mLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�{
VARIABLE_VALUEAdam/dense_8/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_8/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�{
VARIABLE_VALUEAdam/dense_9/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_9/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/dense_10/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_10/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/dense_11/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_11/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE3Adam/module_wrapper_2/batch_normalization_2/gamma/vLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE2Adam/module_wrapper_2/batch_normalization_2/beta/vLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�
serving_default_dense_8_inputPlaceholder*'
_output_shapes
:���������R*
dtype0*
shape:���������R
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_dense_8_inputdense_8/kerneldense_8/biasdense_9/kerneldense_9/biasdense_10/kerneldense_10/bias6module_wrapper_2/batch_normalization_2/moving_variance,module_wrapper_2/batch_normalization_2/gamma2module_wrapper_2/batch_normalization_2/moving_mean+module_wrapper_2/batch_normalization_2/betadense_11/kerneldense_11/bias*
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
GPU2*0J 8� *0
f+R)
'__inference_signature_wrapper_163861662
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
SaveV2SaveV2ShardedFilenameSaveV2/tensor_namesSaveV2/shape_and_slices"dense_8/kernel/Read/ReadVariableOp dense_8/bias/Read/ReadVariableOp"dense_9/kernel/Read/ReadVariableOp dense_9/bias/Read/ReadVariableOp#dense_10/kernel/Read/ReadVariableOp!dense_10/bias/Read/ReadVariableOp#dense_11/kernel/Read/ReadVariableOp!dense_11/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp@module_wrapper_2/batch_normalization_2/gamma/Read/ReadVariableOp?module_wrapper_2/batch_normalization_2/beta/Read/ReadVariableOpFmodule_wrapper_2/batch_normalization_2/moving_mean/Read/ReadVariableOpJmodule_wrapper_2/batch_normalization_2/moving_variance/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOp)Adam/dense_8/kernel/m/Read/ReadVariableOp'Adam/dense_8/bias/m/Read/ReadVariableOp)Adam/dense_9/kernel/m/Read/ReadVariableOp'Adam/dense_9/bias/m/Read/ReadVariableOp*Adam/dense_10/kernel/m/Read/ReadVariableOp(Adam/dense_10/bias/m/Read/ReadVariableOp*Adam/dense_11/kernel/m/Read/ReadVariableOp(Adam/dense_11/bias/m/Read/ReadVariableOpGAdam/module_wrapper_2/batch_normalization_2/gamma/m/Read/ReadVariableOpFAdam/module_wrapper_2/batch_normalization_2/beta/m/Read/ReadVariableOp)Adam/dense_8/kernel/v/Read/ReadVariableOp'Adam/dense_8/bias/v/Read/ReadVariableOp)Adam/dense_9/kernel/v/Read/ReadVariableOp'Adam/dense_9/bias/v/Read/ReadVariableOp*Adam/dense_10/kernel/v/Read/ReadVariableOp(Adam/dense_10/bias/v/Read/ReadVariableOp*Adam/dense_11/kernel/v/Read/ReadVariableOp(Adam/dense_11/bias/v/Read/ReadVariableOpGAdam/module_wrapper_2/batch_normalization_2/gamma/v/Read/ReadVariableOpFAdam/module_wrapper_2/batch_normalization_2/beta/v/Read/ReadVariableOpConst"/device:CPU:0*:
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
\
AssignVariableOpAssignVariableOpdense_8/kernel
Identity_1"/device:CPU:0*
dtype0
U

Identity_2IdentityRestoreV2:1"/device:CPU:0*
T0*
_output_shapes
:
\
AssignVariableOp_1AssignVariableOpdense_8/bias
Identity_2"/device:CPU:0*
dtype0
U

Identity_3IdentityRestoreV2:2"/device:CPU:0*
T0*
_output_shapes
:
^
AssignVariableOp_2AssignVariableOpdense_9/kernel
Identity_3"/device:CPU:0*
dtype0
U

Identity_4IdentityRestoreV2:3"/device:CPU:0*
T0*
_output_shapes
:
\
AssignVariableOp_3AssignVariableOpdense_9/bias
Identity_4"/device:CPU:0*
dtype0
U

Identity_5IdentityRestoreV2:4"/device:CPU:0*
T0*
_output_shapes
:
_
AssignVariableOp_4AssignVariableOpdense_10/kernel
Identity_5"/device:CPU:0*
dtype0
U

Identity_6IdentityRestoreV2:5"/device:CPU:0*
T0*
_output_shapes
:
]
AssignVariableOp_5AssignVariableOpdense_10/bias
Identity_6"/device:CPU:0*
dtype0
U

Identity_7IdentityRestoreV2:6"/device:CPU:0*
T0*
_output_shapes
:
_
AssignVariableOp_6AssignVariableOpdense_11/kernel
Identity_7"/device:CPU:0*
dtype0
U

Identity_8IdentityRestoreV2:7"/device:CPU:0*
T0*
_output_shapes
:
]
AssignVariableOp_7AssignVariableOpdense_11/bias
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
AssignVariableOp_13AssignVariableOp,module_wrapper_2/batch_normalization_2/gammaIdentity_14"/device:CPU:0*
dtype0
W
Identity_15IdentityRestoreV2:14"/device:CPU:0*
T0*
_output_shapes
:
}
AssignVariableOp_14AssignVariableOp+module_wrapper_2/batch_normalization_2/betaIdentity_15"/device:CPU:0*
dtype0
W
Identity_16IdentityRestoreV2:15"/device:CPU:0*
T0*
_output_shapes
:
�
AssignVariableOp_15AssignVariableOp2module_wrapper_2/batch_normalization_2/moving_meanIdentity_16"/device:CPU:0*
dtype0
W
Identity_17IdentityRestoreV2:16"/device:CPU:0*
T0*
_output_shapes
:
�
AssignVariableOp_16AssignVariableOp6module_wrapper_2/batch_normalization_2/moving_varianceIdentity_17"/device:CPU:0*
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
g
AssignVariableOp_23AssignVariableOpAdam/dense_8/kernel/mIdentity_24"/device:CPU:0*
dtype0
W
Identity_25IdentityRestoreV2:24"/device:CPU:0*
T0*
_output_shapes
:
e
AssignVariableOp_24AssignVariableOpAdam/dense_8/bias/mIdentity_25"/device:CPU:0*
dtype0
W
Identity_26IdentityRestoreV2:25"/device:CPU:0*
T0*
_output_shapes
:
g
AssignVariableOp_25AssignVariableOpAdam/dense_9/kernel/mIdentity_26"/device:CPU:0*
dtype0
W
Identity_27IdentityRestoreV2:26"/device:CPU:0*
T0*
_output_shapes
:
e
AssignVariableOp_26AssignVariableOpAdam/dense_9/bias/mIdentity_27"/device:CPU:0*
dtype0
W
Identity_28IdentityRestoreV2:27"/device:CPU:0*
T0*
_output_shapes
:
h
AssignVariableOp_27AssignVariableOpAdam/dense_10/kernel/mIdentity_28"/device:CPU:0*
dtype0
W
Identity_29IdentityRestoreV2:28"/device:CPU:0*
T0*
_output_shapes
:
f
AssignVariableOp_28AssignVariableOpAdam/dense_10/bias/mIdentity_29"/device:CPU:0*
dtype0
W
Identity_30IdentityRestoreV2:29"/device:CPU:0*
T0*
_output_shapes
:
h
AssignVariableOp_29AssignVariableOpAdam/dense_11/kernel/mIdentity_30"/device:CPU:0*
dtype0
W
Identity_31IdentityRestoreV2:30"/device:CPU:0*
T0*
_output_shapes
:
f
AssignVariableOp_30AssignVariableOpAdam/dense_11/bias/mIdentity_31"/device:CPU:0*
dtype0
W
Identity_32IdentityRestoreV2:31"/device:CPU:0*
T0*
_output_shapes
:
�
AssignVariableOp_31AssignVariableOp3Adam/module_wrapper_2/batch_normalization_2/gamma/mIdentity_32"/device:CPU:0*
dtype0
W
Identity_33IdentityRestoreV2:32"/device:CPU:0*
T0*
_output_shapes
:
�
AssignVariableOp_32AssignVariableOp2Adam/module_wrapper_2/batch_normalization_2/beta/mIdentity_33"/device:CPU:0*
dtype0
W
Identity_34IdentityRestoreV2:33"/device:CPU:0*
T0*
_output_shapes
:
g
AssignVariableOp_33AssignVariableOpAdam/dense_8/kernel/vIdentity_34"/device:CPU:0*
dtype0
W
Identity_35IdentityRestoreV2:34"/device:CPU:0*
T0*
_output_shapes
:
e
AssignVariableOp_34AssignVariableOpAdam/dense_8/bias/vIdentity_35"/device:CPU:0*
dtype0
W
Identity_36IdentityRestoreV2:35"/device:CPU:0*
T0*
_output_shapes
:
g
AssignVariableOp_35AssignVariableOpAdam/dense_9/kernel/vIdentity_36"/device:CPU:0*
dtype0
W
Identity_37IdentityRestoreV2:36"/device:CPU:0*
T0*
_output_shapes
:
e
AssignVariableOp_36AssignVariableOpAdam/dense_9/bias/vIdentity_37"/device:CPU:0*
dtype0
W
Identity_38IdentityRestoreV2:37"/device:CPU:0*
T0*
_output_shapes
:
h
AssignVariableOp_37AssignVariableOpAdam/dense_10/kernel/vIdentity_38"/device:CPU:0*
dtype0
W
Identity_39IdentityRestoreV2:38"/device:CPU:0*
T0*
_output_shapes
:
f
AssignVariableOp_38AssignVariableOpAdam/dense_10/bias/vIdentity_39"/device:CPU:0*
dtype0
W
Identity_40IdentityRestoreV2:39"/device:CPU:0*
T0*
_output_shapes
:
h
AssignVariableOp_39AssignVariableOpAdam/dense_11/kernel/vIdentity_40"/device:CPU:0*
dtype0
W
Identity_41IdentityRestoreV2:40"/device:CPU:0*
T0*
_output_shapes
:
f
AssignVariableOp_40AssignVariableOpAdam/dense_11/bias/vIdentity_41"/device:CPU:0*
dtype0
W
Identity_42IdentityRestoreV2:41"/device:CPU:0*
T0*
_output_shapes
:
�
AssignVariableOp_41AssignVariableOp3Adam/module_wrapper_2/batch_normalization_2/gamma/vIdentity_42"/device:CPU:0*
dtype0
W
Identity_43IdentityRestoreV2:42"/device:CPU:0*
T0*
_output_shapes
:
�
AssignVariableOp_42AssignVariableOp2Adam/module_wrapper_2/batch_normalization_2/beta/vIdentity_43"/device:CPU:0*
dtype0

NoOp_1NoOp"/device:CPU:0
�
Identity_44Identitysaver_filename^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp_1"/device:CPU:0*
T0*
_output_shapes
: ��
�$
�
9__inference_batch_normalization_2_layer_call_fn_163862039

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
�C
�
0__inference_sequential_2_layer_call_fn_163861461

inputs8
&dense_8_matmul_readvariableop_resource:Rf5
'dense_8_biasadd_readvariableop_resource:f8
&dense_9_matmul_readvariableop_resource:ff5
'dense_9_biasadd_readvariableop_resource:f9
'dense_10_matmul_readvariableop_resource:ff6
(dense_10_biasadd_readvariableop_resource:fV
Hmodule_wrapper_2_batch_normalization_2_batchnorm_readvariableop_resource:fZ
Lmodule_wrapper_2_batch_normalization_2_batchnorm_mul_readvariableop_resource:fX
Jmodule_wrapper_2_batch_normalization_2_batchnorm_readvariableop_1_resource:fX
Jmodule_wrapper_2_batch_normalization_2_batchnorm_readvariableop_2_resource:f:
'dense_11_matmul_readvariableop_resource:	f�7
(dense_11_biasadd_readvariableop_resource:	�
identity��dense_10/BiasAdd/ReadVariableOp�dense_10/MatMul/ReadVariableOp�dense_11/BiasAdd/ReadVariableOp�dense_11/MatMul/ReadVariableOp�dense_8/BiasAdd/ReadVariableOp�dense_8/MatMul/ReadVariableOp�dense_9/BiasAdd/ReadVariableOp�dense_9/MatMul/ReadVariableOp�?module_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp�Amodule_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp_1�Amodule_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp_2�Cmodule_wrapper_2/batch_normalization_2/batchnorm/mul/ReadVariableOp�
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes

:Rf*
dtype0y
dense_8/MatMulMatMulinputs%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f�
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:f*
dtype0�
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f`
dense_8/ReluReludense_8/BiasAdd:output:0*
T0*'
_output_shapes
:���������f�
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes

:ff*
dtype0�
dense_9/MatMulMatMuldense_8/Relu:activations:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f�
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:f*
dtype0�
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f`
dense_9/ReluReludense_9/BiasAdd:output:0*
T0*'
_output_shapes
:���������f�
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource*
_output_shapes

:ff*
dtype0�
dense_10/MatMulMatMuldense_9/Relu:activations:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f�
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes
:f*
dtype0�
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������fb
dense_10/ReluReludense_10/BiasAdd:output:0*
T0*'
_output_shapes
:���������f�
?module_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOpReadVariableOpHmodule_wrapper_2_batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:f*
dtype0{
6module_wrapper_2/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
4module_wrapper_2/batch_normalization_2/batchnorm/addAddV2Gmodule_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp:value:0?module_wrapper_2/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:f�
6module_wrapper_2/batch_normalization_2/batchnorm/RsqrtRsqrt8module_wrapper_2/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:f�
Cmodule_wrapper_2/batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpLmodule_wrapper_2_batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:f*
dtype0�
4module_wrapper_2/batch_normalization_2/batchnorm/mulMul:module_wrapper_2/batch_normalization_2/batchnorm/Rsqrt:y:0Kmodule_wrapper_2/batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:f�
6module_wrapper_2/batch_normalization_2/batchnorm/mul_1Muldense_10/Relu:activations:08module_wrapper_2/batch_normalization_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������f�
Amodule_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOpJmodule_wrapper_2_batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes
:f*
dtype0�
6module_wrapper_2/batch_normalization_2/batchnorm/mul_2MulImodule_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp_1:value:08module_wrapper_2/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:f�
Amodule_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOpJmodule_wrapper_2_batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes
:f*
dtype0�
4module_wrapper_2/batch_normalization_2/batchnorm/subSubImodule_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp_2:value:0:module_wrapper_2/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:f�
6module_wrapper_2/batch_normalization_2/batchnorm/add_1AddV2:module_wrapper_2/batch_normalization_2/batchnorm/mul_1:z:08module_wrapper_2/batch_normalization_2/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������f�
dropout_2/IdentityIdentity:module_wrapper_2/batch_normalization_2/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������f�
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes
:	f�*
dtype0�
dense_11/MatMulMatMuldropout_2/Identity:output:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������i
IdentityIdentitydense_11/BiasAdd:output:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp@^module_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOpB^module_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp_1B^module_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp_2D^module_wrapper_2/batch_normalization_2/batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������R: : : : : : : : : : : : 2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp2�
?module_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp?module_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp2�
Amodule_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp_1Amodule_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp_12�
Amodule_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp_2Amodule_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp_22�
Cmodule_wrapper_2/batch_normalization_2/batchnorm/mul/ReadVariableOpCmodule_wrapper_2/batch_normalization_2/batchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������R
 
_user_specified_nameinputs
�
�
4__inference_module_wrapper_2_layer_call_fn_163861748

args_0E
7batch_normalization_2_batchnorm_readvariableop_resource:fI
;batch_normalization_2_batchnorm_mul_readvariableop_resource:fG
9batch_normalization_2_batchnorm_readvariableop_1_resource:fG
9batch_normalization_2_batchnorm_readvariableop_2_resource:f
identity��.batch_normalization_2/batchnorm/ReadVariableOp�0batch_normalization_2/batchnorm/ReadVariableOp_1�0batch_normalization_2/batchnorm/ReadVariableOp_2�2batch_normalization_2/batchnorm/mul/ReadVariableOp�
.batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:f*
dtype0j
%batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
#batch_normalization_2/batchnorm/addAddV26batch_normalization_2/batchnorm/ReadVariableOp:value:0.batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:f|
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:f�
2batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:f*
dtype0�
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:0:batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:f�
%batch_normalization_2/batchnorm/mul_1Mulargs_0'batch_normalization_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������f�
0batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes
:f*
dtype0�
%batch_normalization_2/batchnorm/mul_2Mul8batch_normalization_2/batchnorm/ReadVariableOp_1:value:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:f�
0batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes
:f*
dtype0�
#batch_normalization_2/batchnorm/subSub8batch_normalization_2/batchnorm/ReadVariableOp_2:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:f�
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������fx
IdentityIdentity)batch_normalization_2/batchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������f�
NoOpNoOp/^batch_normalization_2/batchnorm/ReadVariableOp1^batch_normalization_2/batchnorm/ReadVariableOp_11^batch_normalization_2/batchnorm/ReadVariableOp_23^batch_normalization_2/batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������f: : : : 2`
.batch_normalization_2/batchnorm/ReadVariableOp.batch_normalization_2/batchnorm/ReadVariableOp2d
0batch_normalization_2/batchnorm/ReadVariableOp_10batch_normalization_2/batchnorm/ReadVariableOp_12d
0batch_normalization_2/batchnorm/ReadVariableOp_20batch_normalization_2/batchnorm/ReadVariableOp_22h
2batch_normalization_2/batchnorm/mul/ReadVariableOp2batch_normalization_2/batchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������f
 
_user_specified_nameargs_0
�

�
'__inference_signature_wrapper_163861662
dense_8_input
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
StatefulPartitionedCallStatefulPartitionedCalldense_8_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
GPU2*0J 8� *-
f(R&
$__inference__wrapped_model_163860745p
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
StatefulPartitionedCallStatefulPartitionedCall:V R
'
_output_shapes
:���������R
'
_user_specified_namedense_8_input
�5
�
O__inference_module_wrapper_2_layer_call_and_return_conditional_losses_163861836

args_0K
=batch_normalization_2_assignmovingavg_readvariableop_resource:fM
?batch_normalization_2_assignmovingavg_1_readvariableop_resource:fI
;batch_normalization_2_batchnorm_mul_readvariableop_resource:fE
7batch_normalization_2_batchnorm_readvariableop_resource:f
identity��%batch_normalization_2/AssignMovingAvg�4batch_normalization_2/AssignMovingAvg/ReadVariableOp�'batch_normalization_2/AssignMovingAvg_1�6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp�.batch_normalization_2/batchnorm/ReadVariableOp�2batch_normalization_2/batchnorm/mul/ReadVariableOp~
4batch_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
"batch_normalization_2/moments/meanMeanargs_0=batch_normalization_2/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:f*
	keep_dims(�
*batch_normalization_2/moments/StopGradientStopGradient+batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes

:f�
/batch_normalization_2/moments/SquaredDifferenceSquaredDifferenceargs_03batch_normalization_2/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������f�
8batch_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
&batch_normalization_2/moments/varianceMean3batch_normalization_2/moments/SquaredDifference:z:0Abatch_normalization_2/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:f*
	keep_dims(�
%batch_normalization_2/moments/SqueezeSqueeze+batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes
:f*
squeeze_dims
 �
'batch_normalization_2/moments/Squeeze_1Squeeze/batch_normalization_2/moments/variance:output:0*
T0*
_output_shapes
:f*
squeeze_dims
 p
+batch_normalization_2/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
4batch_normalization_2/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_2_assignmovingavg_readvariableop_resource*
_output_shapes
:f*
dtype0�
)batch_normalization_2/AssignMovingAvg/subSub<batch_normalization_2/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_2/moments/Squeeze:output:0*
T0*
_output_shapes
:f�
)batch_normalization_2/AssignMovingAvg/mulMul-batch_normalization_2/AssignMovingAvg/sub:z:04batch_normalization_2/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:f�
%batch_normalization_2/AssignMovingAvgAssignSubVariableOp=batch_normalization_2_assignmovingavg_readvariableop_resource-batch_normalization_2/AssignMovingAvg/mul:z:05^batch_normalization_2/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0r
-batch_normalization_2/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_2_assignmovingavg_1_readvariableop_resource*
_output_shapes
:f*
dtype0�
+batch_normalization_2/AssignMovingAvg_1/subSub>batch_normalization_2/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_2/moments/Squeeze_1:output:0*
T0*
_output_shapes
:f�
+batch_normalization_2/AssignMovingAvg_1/mulMul/batch_normalization_2/AssignMovingAvg_1/sub:z:06batch_normalization_2/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:f�
'batch_normalization_2/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_2_assignmovingavg_1_readvariableop_resource/batch_normalization_2/AssignMovingAvg_1/mul:z:07^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0j
%batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
#batch_normalization_2/batchnorm/addAddV20batch_normalization_2/moments/Squeeze_1:output:0.batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:f|
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:f�
2batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:f*
dtype0�
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:0:batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:f�
%batch_normalization_2/batchnorm/mul_1Mulargs_0'batch_normalization_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������f�
%batch_normalization_2/batchnorm/mul_2Mul.batch_normalization_2/moments/Squeeze:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:f�
.batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:f*
dtype0�
#batch_normalization_2/batchnorm/subSub6batch_normalization_2/batchnorm/ReadVariableOp:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:f�
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������fx
IdentityIdentity)batch_normalization_2/batchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������f�
NoOpNoOp&^batch_normalization_2/AssignMovingAvg5^batch_normalization_2/AssignMovingAvg/ReadVariableOp(^batch_normalization_2/AssignMovingAvg_17^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_2/batchnorm/ReadVariableOp3^batch_normalization_2/batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������f: : : : 2N
%batch_normalization_2/AssignMovingAvg%batch_normalization_2/AssignMovingAvg2l
4batch_normalization_2/AssignMovingAvg/ReadVariableOp4batch_normalization_2/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_2/AssignMovingAvg_1'batch_normalization_2/AssignMovingAvg_12p
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_2/batchnorm/ReadVariableOp.batch_normalization_2/batchnorm/ReadVariableOp2h
2batch_normalization_2/batchnorm/mul/ReadVariableOp2batch_normalization_2/batchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������f
 
_user_specified_nameargs_0
�
d
H__inference_dropout_2_layer_call_and_return_conditional_losses_163861854

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
�b
�
K__inference_sequential_2_layer_call_and_return_conditional_losses_163861631

inputs8
&dense_8_matmul_readvariableop_resource:Rf5
'dense_8_biasadd_readvariableop_resource:f8
&dense_9_matmul_readvariableop_resource:ff5
'dense_9_biasadd_readvariableop_resource:f9
'dense_10_matmul_readvariableop_resource:ff6
(dense_10_biasadd_readvariableop_resource:f\
Nmodule_wrapper_2_batch_normalization_2_assignmovingavg_readvariableop_resource:f^
Pmodule_wrapper_2_batch_normalization_2_assignmovingavg_1_readvariableop_resource:fZ
Lmodule_wrapper_2_batch_normalization_2_batchnorm_mul_readvariableop_resource:fV
Hmodule_wrapper_2_batch_normalization_2_batchnorm_readvariableop_resource:f:
'dense_11_matmul_readvariableop_resource:	f�7
(dense_11_biasadd_readvariableop_resource:	�
identity��dense_10/BiasAdd/ReadVariableOp�dense_10/MatMul/ReadVariableOp�dense_11/BiasAdd/ReadVariableOp�dense_11/MatMul/ReadVariableOp�dense_8/BiasAdd/ReadVariableOp�dense_8/MatMul/ReadVariableOp�dense_9/BiasAdd/ReadVariableOp�dense_9/MatMul/ReadVariableOp�6module_wrapper_2/batch_normalization_2/AssignMovingAvg�Emodule_wrapper_2/batch_normalization_2/AssignMovingAvg/ReadVariableOp�8module_wrapper_2/batch_normalization_2/AssignMovingAvg_1�Gmodule_wrapper_2/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp�?module_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp�Cmodule_wrapper_2/batch_normalization_2/batchnorm/mul/ReadVariableOp�
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes

:Rf*
dtype0y
dense_8/MatMulMatMulinputs%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f�
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:f*
dtype0�
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f`
dense_8/ReluReludense_8/BiasAdd:output:0*
T0*'
_output_shapes
:���������f�
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes

:ff*
dtype0�
dense_9/MatMulMatMuldense_8/Relu:activations:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f�
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:f*
dtype0�
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f`
dense_9/ReluReludense_9/BiasAdd:output:0*
T0*'
_output_shapes
:���������f�
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource*
_output_shapes

:ff*
dtype0�
dense_10/MatMulMatMuldense_9/Relu:activations:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f�
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes
:f*
dtype0�
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������fb
dense_10/ReluReludense_10/BiasAdd:output:0*
T0*'
_output_shapes
:���������f�
Emodule_wrapper_2/batch_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
3module_wrapper_2/batch_normalization_2/moments/meanMeandense_10/Relu:activations:0Nmodule_wrapper_2/batch_normalization_2/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:f*
	keep_dims(�
;module_wrapper_2/batch_normalization_2/moments/StopGradientStopGradient<module_wrapper_2/batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes

:f�
@module_wrapper_2/batch_normalization_2/moments/SquaredDifferenceSquaredDifferencedense_10/Relu:activations:0Dmodule_wrapper_2/batch_normalization_2/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������f�
Imodule_wrapper_2/batch_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
7module_wrapper_2/batch_normalization_2/moments/varianceMeanDmodule_wrapper_2/batch_normalization_2/moments/SquaredDifference:z:0Rmodule_wrapper_2/batch_normalization_2/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:f*
	keep_dims(�
6module_wrapper_2/batch_normalization_2/moments/SqueezeSqueeze<module_wrapper_2/batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes
:f*
squeeze_dims
 �
8module_wrapper_2/batch_normalization_2/moments/Squeeze_1Squeeze@module_wrapper_2/batch_normalization_2/moments/variance:output:0*
T0*
_output_shapes
:f*
squeeze_dims
 �
<module_wrapper_2/batch_normalization_2/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
Emodule_wrapper_2/batch_normalization_2/AssignMovingAvg/ReadVariableOpReadVariableOpNmodule_wrapper_2_batch_normalization_2_assignmovingavg_readvariableop_resource*
_output_shapes
:f*
dtype0�
:module_wrapper_2/batch_normalization_2/AssignMovingAvg/subSubMmodule_wrapper_2/batch_normalization_2/AssignMovingAvg/ReadVariableOp:value:0?module_wrapper_2/batch_normalization_2/moments/Squeeze:output:0*
T0*
_output_shapes
:f�
:module_wrapper_2/batch_normalization_2/AssignMovingAvg/mulMul>module_wrapper_2/batch_normalization_2/AssignMovingAvg/sub:z:0Emodule_wrapper_2/batch_normalization_2/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:f�
6module_wrapper_2/batch_normalization_2/AssignMovingAvgAssignSubVariableOpNmodule_wrapper_2_batch_normalization_2_assignmovingavg_readvariableop_resource>module_wrapper_2/batch_normalization_2/AssignMovingAvg/mul:z:0F^module_wrapper_2/batch_normalization_2/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0�
>module_wrapper_2/batch_normalization_2/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
Gmodule_wrapper_2/batch_normalization_2/AssignMovingAvg_1/ReadVariableOpReadVariableOpPmodule_wrapper_2_batch_normalization_2_assignmovingavg_1_readvariableop_resource*
_output_shapes
:f*
dtype0�
<module_wrapper_2/batch_normalization_2/AssignMovingAvg_1/subSubOmodule_wrapper_2/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp:value:0Amodule_wrapper_2/batch_normalization_2/moments/Squeeze_1:output:0*
T0*
_output_shapes
:f�
<module_wrapper_2/batch_normalization_2/AssignMovingAvg_1/mulMul@module_wrapper_2/batch_normalization_2/AssignMovingAvg_1/sub:z:0Gmodule_wrapper_2/batch_normalization_2/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:f�
8module_wrapper_2/batch_normalization_2/AssignMovingAvg_1AssignSubVariableOpPmodule_wrapper_2_batch_normalization_2_assignmovingavg_1_readvariableop_resource@module_wrapper_2/batch_normalization_2/AssignMovingAvg_1/mul:z:0H^module_wrapper_2/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0{
6module_wrapper_2/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
4module_wrapper_2/batch_normalization_2/batchnorm/addAddV2Amodule_wrapper_2/batch_normalization_2/moments/Squeeze_1:output:0?module_wrapper_2/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:f�
6module_wrapper_2/batch_normalization_2/batchnorm/RsqrtRsqrt8module_wrapper_2/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:f�
Cmodule_wrapper_2/batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpLmodule_wrapper_2_batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:f*
dtype0�
4module_wrapper_2/batch_normalization_2/batchnorm/mulMul:module_wrapper_2/batch_normalization_2/batchnorm/Rsqrt:y:0Kmodule_wrapper_2/batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:f�
6module_wrapper_2/batch_normalization_2/batchnorm/mul_1Muldense_10/Relu:activations:08module_wrapper_2/batch_normalization_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������f�
6module_wrapper_2/batch_normalization_2/batchnorm/mul_2Mul?module_wrapper_2/batch_normalization_2/moments/Squeeze:output:08module_wrapper_2/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:f�
?module_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOpReadVariableOpHmodule_wrapper_2_batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:f*
dtype0�
4module_wrapper_2/batch_normalization_2/batchnorm/subSubGmodule_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp:value:0:module_wrapper_2/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:f�
6module_wrapper_2/batch_normalization_2/batchnorm/add_1AddV2:module_wrapper_2/batch_normalization_2/batchnorm/mul_1:z:08module_wrapper_2/batch_normalization_2/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������f�
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes
:	f�*
dtype0�
dense_11/MatMulMatMul:module_wrapper_2/batch_normalization_2/batchnorm/add_1:z:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������i
IdentityIdentitydense_11/BiasAdd:output:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp7^module_wrapper_2/batch_normalization_2/AssignMovingAvgF^module_wrapper_2/batch_normalization_2/AssignMovingAvg/ReadVariableOp9^module_wrapper_2/batch_normalization_2/AssignMovingAvg_1H^module_wrapper_2/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp@^module_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOpD^module_wrapper_2/batch_normalization_2/batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������R: : : : : : : : : : : : 2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp2p
6module_wrapper_2/batch_normalization_2/AssignMovingAvg6module_wrapper_2/batch_normalization_2/AssignMovingAvg2�
Emodule_wrapper_2/batch_normalization_2/AssignMovingAvg/ReadVariableOpEmodule_wrapper_2/batch_normalization_2/AssignMovingAvg/ReadVariableOp2t
8module_wrapper_2/batch_normalization_2/AssignMovingAvg_18module_wrapper_2/batch_normalization_2/AssignMovingAvg_12�
Gmodule_wrapper_2/batch_normalization_2/AssignMovingAvg_1/ReadVariableOpGmodule_wrapper_2/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp2�
?module_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp?module_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp2�
Cmodule_wrapper_2/batch_normalization_2/batchnorm/mul/ReadVariableOpCmodule_wrapper_2/batch_normalization_2/batchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������R
 
_user_specified_nameinputs
�C
�
K__inference_sequential_2_layer_call_and_return_conditional_losses_163861346
dense_8_input8
&dense_8_matmul_readvariableop_resource:Rf5
'dense_8_biasadd_readvariableop_resource:f8
&dense_9_matmul_readvariableop_resource:ff5
'dense_9_biasadd_readvariableop_resource:f9
'dense_10_matmul_readvariableop_resource:ff6
(dense_10_biasadd_readvariableop_resource:fV
Hmodule_wrapper_2_batch_normalization_2_batchnorm_readvariableop_resource:fZ
Lmodule_wrapper_2_batch_normalization_2_batchnorm_mul_readvariableop_resource:fX
Jmodule_wrapper_2_batch_normalization_2_batchnorm_readvariableop_1_resource:fX
Jmodule_wrapper_2_batch_normalization_2_batchnorm_readvariableop_2_resource:f:
'dense_11_matmul_readvariableop_resource:	f�7
(dense_11_biasadd_readvariableop_resource:	�
identity��dense_10/BiasAdd/ReadVariableOp�dense_10/MatMul/ReadVariableOp�dense_11/BiasAdd/ReadVariableOp�dense_11/MatMul/ReadVariableOp�dense_8/BiasAdd/ReadVariableOp�dense_8/MatMul/ReadVariableOp�dense_9/BiasAdd/ReadVariableOp�dense_9/MatMul/ReadVariableOp�?module_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp�Amodule_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp_1�Amodule_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp_2�Cmodule_wrapper_2/batch_normalization_2/batchnorm/mul/ReadVariableOp�
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes

:Rf*
dtype0�
dense_8/MatMulMatMuldense_8_input%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f�
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:f*
dtype0�
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f`
dense_8/ReluReludense_8/BiasAdd:output:0*
T0*'
_output_shapes
:���������f�
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes

:ff*
dtype0�
dense_9/MatMulMatMuldense_8/Relu:activations:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f�
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:f*
dtype0�
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f`
dense_9/ReluReludense_9/BiasAdd:output:0*
T0*'
_output_shapes
:���������f�
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource*
_output_shapes

:ff*
dtype0�
dense_10/MatMulMatMuldense_9/Relu:activations:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f�
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes
:f*
dtype0�
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������fb
dense_10/ReluReludense_10/BiasAdd:output:0*
T0*'
_output_shapes
:���������f�
?module_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOpReadVariableOpHmodule_wrapper_2_batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:f*
dtype0{
6module_wrapper_2/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
4module_wrapper_2/batch_normalization_2/batchnorm/addAddV2Gmodule_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp:value:0?module_wrapper_2/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:f�
6module_wrapper_2/batch_normalization_2/batchnorm/RsqrtRsqrt8module_wrapper_2/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:f�
Cmodule_wrapper_2/batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpLmodule_wrapper_2_batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:f*
dtype0�
4module_wrapper_2/batch_normalization_2/batchnorm/mulMul:module_wrapper_2/batch_normalization_2/batchnorm/Rsqrt:y:0Kmodule_wrapper_2/batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:f�
6module_wrapper_2/batch_normalization_2/batchnorm/mul_1Muldense_10/Relu:activations:08module_wrapper_2/batch_normalization_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������f�
Amodule_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOpJmodule_wrapper_2_batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes
:f*
dtype0�
6module_wrapper_2/batch_normalization_2/batchnorm/mul_2MulImodule_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp_1:value:08module_wrapper_2/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:f�
Amodule_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOpJmodule_wrapper_2_batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes
:f*
dtype0�
4module_wrapper_2/batch_normalization_2/batchnorm/subSubImodule_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp_2:value:0:module_wrapper_2/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:f�
6module_wrapper_2/batch_normalization_2/batchnorm/add_1AddV2:module_wrapper_2/batch_normalization_2/batchnorm/mul_1:z:08module_wrapper_2/batch_normalization_2/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������f�
dropout_2/IdentityIdentity:module_wrapper_2/batch_normalization_2/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������f�
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes
:	f�*
dtype0�
dense_11/MatMulMatMuldropout_2/Identity:output:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������i
IdentityIdentitydense_11/BiasAdd:output:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp@^module_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOpB^module_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp_1B^module_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp_2D^module_wrapper_2/batch_normalization_2/batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������R: : : : : : : : : : : : 2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp2�
?module_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp?module_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp2�
Amodule_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp_1Amodule_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp_12�
Amodule_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp_2Amodule_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp_22�
Cmodule_wrapper_2/batch_normalization_2/batchnorm/mul/ReadVariableOpCmodule_wrapper_2/batch_normalization_2/batchnorm/mul/ReadVariableOp:V R
'
_output_shapes
:���������R
'
_user_specified_namedense_8_input
�
�
9__inference_batch_normalization_2_layer_call_fn_163862005

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
G__inference_dense_10_layer_call_and_return_conditional_losses_163861728

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
�C
�
K__inference_sequential_2_layer_call_and_return_conditional_losses_163861570

inputs8
&dense_8_matmul_readvariableop_resource:Rf5
'dense_8_biasadd_readvariableop_resource:f8
&dense_9_matmul_readvariableop_resource:ff5
'dense_9_biasadd_readvariableop_resource:f9
'dense_10_matmul_readvariableop_resource:ff6
(dense_10_biasadd_readvariableop_resource:fV
Hmodule_wrapper_2_batch_normalization_2_batchnorm_readvariableop_resource:fZ
Lmodule_wrapper_2_batch_normalization_2_batchnorm_mul_readvariableop_resource:fX
Jmodule_wrapper_2_batch_normalization_2_batchnorm_readvariableop_1_resource:fX
Jmodule_wrapper_2_batch_normalization_2_batchnorm_readvariableop_2_resource:f:
'dense_11_matmul_readvariableop_resource:	f�7
(dense_11_biasadd_readvariableop_resource:	�
identity��dense_10/BiasAdd/ReadVariableOp�dense_10/MatMul/ReadVariableOp�dense_11/BiasAdd/ReadVariableOp�dense_11/MatMul/ReadVariableOp�dense_8/BiasAdd/ReadVariableOp�dense_8/MatMul/ReadVariableOp�dense_9/BiasAdd/ReadVariableOp�dense_9/MatMul/ReadVariableOp�?module_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp�Amodule_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp_1�Amodule_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp_2�Cmodule_wrapper_2/batch_normalization_2/batchnorm/mul/ReadVariableOp�
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes

:Rf*
dtype0y
dense_8/MatMulMatMulinputs%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f�
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:f*
dtype0�
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f`
dense_8/ReluReludense_8/BiasAdd:output:0*
T0*'
_output_shapes
:���������f�
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes

:ff*
dtype0�
dense_9/MatMulMatMuldense_8/Relu:activations:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f�
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:f*
dtype0�
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f`
dense_9/ReluReludense_9/BiasAdd:output:0*
T0*'
_output_shapes
:���������f�
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource*
_output_shapes

:ff*
dtype0�
dense_10/MatMulMatMuldense_9/Relu:activations:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f�
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes
:f*
dtype0�
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������fb
dense_10/ReluReludense_10/BiasAdd:output:0*
T0*'
_output_shapes
:���������f�
?module_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOpReadVariableOpHmodule_wrapper_2_batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:f*
dtype0{
6module_wrapper_2/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
4module_wrapper_2/batch_normalization_2/batchnorm/addAddV2Gmodule_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp:value:0?module_wrapper_2/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:f�
6module_wrapper_2/batch_normalization_2/batchnorm/RsqrtRsqrt8module_wrapper_2/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:f�
Cmodule_wrapper_2/batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpLmodule_wrapper_2_batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:f*
dtype0�
4module_wrapper_2/batch_normalization_2/batchnorm/mulMul:module_wrapper_2/batch_normalization_2/batchnorm/Rsqrt:y:0Kmodule_wrapper_2/batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:f�
6module_wrapper_2/batch_normalization_2/batchnorm/mul_1Muldense_10/Relu:activations:08module_wrapper_2/batch_normalization_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������f�
Amodule_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOpJmodule_wrapper_2_batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes
:f*
dtype0�
6module_wrapper_2/batch_normalization_2/batchnorm/mul_2MulImodule_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp_1:value:08module_wrapper_2/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:f�
Amodule_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOpJmodule_wrapper_2_batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes
:f*
dtype0�
4module_wrapper_2/batch_normalization_2/batchnorm/subSubImodule_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp_2:value:0:module_wrapper_2/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:f�
6module_wrapper_2/batch_normalization_2/batchnorm/add_1AddV2:module_wrapper_2/batch_normalization_2/batchnorm/mul_1:z:08module_wrapper_2/batch_normalization_2/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������f�
dropout_2/IdentityIdentity:module_wrapper_2/batch_normalization_2/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������f�
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes
:	f�*
dtype0�
dense_11/MatMulMatMuldropout_2/Identity:output:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������i
IdentityIdentitydense_11/BiasAdd:output:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp@^module_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOpB^module_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp_1B^module_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp_2D^module_wrapper_2/batch_normalization_2/batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������R: : : : : : : : : : : : 2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp2�
?module_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp?module_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp2�
Amodule_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp_1Amodule_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp_12�
Amodule_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp_2Amodule_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp_22�
Cmodule_wrapper_2/batch_normalization_2/batchnorm/mul/ReadVariableOpCmodule_wrapper_2/batch_normalization_2/batchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������R
 
_user_specified_nameinputs
�C
�
0__inference_sequential_2_layer_call_fn_163860794
dense_8_input8
&dense_8_matmul_readvariableop_resource:Rf5
'dense_8_biasadd_readvariableop_resource:f8
&dense_9_matmul_readvariableop_resource:ff5
'dense_9_biasadd_readvariableop_resource:f9
'dense_10_matmul_readvariableop_resource:ff6
(dense_10_biasadd_readvariableop_resource:fV
Hmodule_wrapper_2_batch_normalization_2_batchnorm_readvariableop_resource:fZ
Lmodule_wrapper_2_batch_normalization_2_batchnorm_mul_readvariableop_resource:fX
Jmodule_wrapper_2_batch_normalization_2_batchnorm_readvariableop_1_resource:fX
Jmodule_wrapper_2_batch_normalization_2_batchnorm_readvariableop_2_resource:f:
'dense_11_matmul_readvariableop_resource:	f�7
(dense_11_biasadd_readvariableop_resource:	�
identity��dense_10/BiasAdd/ReadVariableOp�dense_10/MatMul/ReadVariableOp�dense_11/BiasAdd/ReadVariableOp�dense_11/MatMul/ReadVariableOp�dense_8/BiasAdd/ReadVariableOp�dense_8/MatMul/ReadVariableOp�dense_9/BiasAdd/ReadVariableOp�dense_9/MatMul/ReadVariableOp�?module_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp�Amodule_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp_1�Amodule_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp_2�Cmodule_wrapper_2/batch_normalization_2/batchnorm/mul/ReadVariableOp�
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes

:Rf*
dtype0�
dense_8/MatMulMatMuldense_8_input%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f�
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:f*
dtype0�
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f`
dense_8/ReluReludense_8/BiasAdd:output:0*
T0*'
_output_shapes
:���������f�
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes

:ff*
dtype0�
dense_9/MatMulMatMuldense_8/Relu:activations:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f�
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:f*
dtype0�
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f`
dense_9/ReluReludense_9/BiasAdd:output:0*
T0*'
_output_shapes
:���������f�
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource*
_output_shapes

:ff*
dtype0�
dense_10/MatMulMatMuldense_9/Relu:activations:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f�
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes
:f*
dtype0�
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������fb
dense_10/ReluReludense_10/BiasAdd:output:0*
T0*'
_output_shapes
:���������f�
?module_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOpReadVariableOpHmodule_wrapper_2_batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:f*
dtype0{
6module_wrapper_2/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
4module_wrapper_2/batch_normalization_2/batchnorm/addAddV2Gmodule_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp:value:0?module_wrapper_2/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:f�
6module_wrapper_2/batch_normalization_2/batchnorm/RsqrtRsqrt8module_wrapper_2/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:f�
Cmodule_wrapper_2/batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpLmodule_wrapper_2_batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:f*
dtype0�
4module_wrapper_2/batch_normalization_2/batchnorm/mulMul:module_wrapper_2/batch_normalization_2/batchnorm/Rsqrt:y:0Kmodule_wrapper_2/batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:f�
6module_wrapper_2/batch_normalization_2/batchnorm/mul_1Muldense_10/Relu:activations:08module_wrapper_2/batch_normalization_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������f�
Amodule_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOpJmodule_wrapper_2_batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes
:f*
dtype0�
6module_wrapper_2/batch_normalization_2/batchnorm/mul_2MulImodule_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp_1:value:08module_wrapper_2/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:f�
Amodule_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOpJmodule_wrapper_2_batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes
:f*
dtype0�
4module_wrapper_2/batch_normalization_2/batchnorm/subSubImodule_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp_2:value:0:module_wrapper_2/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:f�
6module_wrapper_2/batch_normalization_2/batchnorm/add_1AddV2:module_wrapper_2/batch_normalization_2/batchnorm/mul_1:z:08module_wrapper_2/batch_normalization_2/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������f�
dropout_2/IdentityIdentity:module_wrapper_2/batch_normalization_2/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������f�
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes
:	f�*
dtype0�
dense_11/MatMulMatMuldropout_2/Identity:output:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������i
IdentityIdentitydense_11/BiasAdd:output:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp@^module_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOpB^module_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp_1B^module_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp_2D^module_wrapper_2/batch_normalization_2/batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������R: : : : : : : : : : : : 2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp2�
?module_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp?module_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp2�
Amodule_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp_1Amodule_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp_12�
Amodule_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp_2Amodule_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp_22�
Cmodule_wrapper_2/batch_normalization_2/batchnorm/mul/ReadVariableOpCmodule_wrapper_2/batch_normalization_2/batchnorm/mul/ReadVariableOp:V R
'
_output_shapes
:���������R
'
_user_specified_namedense_8_input
�

�
,__inference_dense_10_layer_call_fn_163861717

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
�

�
F__inference_dense_9_layer_call_and_return_conditional_losses_163861706

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
�

�
F__inference_dense_8_layer_call_and_return_conditional_losses_163861684

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
0__inference_sequential_2_layer_call_fn_163861298
dense_8_input8
&dense_8_matmul_readvariableop_resource:Rf5
'dense_8_biasadd_readvariableop_resource:f8
&dense_9_matmul_readvariableop_resource:ff5
'dense_9_biasadd_readvariableop_resource:f9
'dense_10_matmul_readvariableop_resource:ff6
(dense_10_biasadd_readvariableop_resource:f\
Nmodule_wrapper_2_batch_normalization_2_assignmovingavg_readvariableop_resource:f^
Pmodule_wrapper_2_batch_normalization_2_assignmovingavg_1_readvariableop_resource:fZ
Lmodule_wrapper_2_batch_normalization_2_batchnorm_mul_readvariableop_resource:fV
Hmodule_wrapper_2_batch_normalization_2_batchnorm_readvariableop_resource:f:
'dense_11_matmul_readvariableop_resource:	f�7
(dense_11_biasadd_readvariableop_resource:	�
identity��dense_10/BiasAdd/ReadVariableOp�dense_10/MatMul/ReadVariableOp�dense_11/BiasAdd/ReadVariableOp�dense_11/MatMul/ReadVariableOp�dense_8/BiasAdd/ReadVariableOp�dense_8/MatMul/ReadVariableOp�dense_9/BiasAdd/ReadVariableOp�dense_9/MatMul/ReadVariableOp�6module_wrapper_2/batch_normalization_2/AssignMovingAvg�Emodule_wrapper_2/batch_normalization_2/AssignMovingAvg/ReadVariableOp�8module_wrapper_2/batch_normalization_2/AssignMovingAvg_1�Gmodule_wrapper_2/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp�?module_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp�Cmodule_wrapper_2/batch_normalization_2/batchnorm/mul/ReadVariableOp�
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes

:Rf*
dtype0�
dense_8/MatMulMatMuldense_8_input%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f�
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:f*
dtype0�
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f`
dense_8/ReluReludense_8/BiasAdd:output:0*
T0*'
_output_shapes
:���������f�
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes

:ff*
dtype0�
dense_9/MatMulMatMuldense_8/Relu:activations:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f�
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:f*
dtype0�
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f`
dense_9/ReluReludense_9/BiasAdd:output:0*
T0*'
_output_shapes
:���������f�
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource*
_output_shapes

:ff*
dtype0�
dense_10/MatMulMatMuldense_9/Relu:activations:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f�
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes
:f*
dtype0�
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������fb
dense_10/ReluReludense_10/BiasAdd:output:0*
T0*'
_output_shapes
:���������f�
Emodule_wrapper_2/batch_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
3module_wrapper_2/batch_normalization_2/moments/meanMeandense_10/Relu:activations:0Nmodule_wrapper_2/batch_normalization_2/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:f*
	keep_dims(�
;module_wrapper_2/batch_normalization_2/moments/StopGradientStopGradient<module_wrapper_2/batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes

:f�
@module_wrapper_2/batch_normalization_2/moments/SquaredDifferenceSquaredDifferencedense_10/Relu:activations:0Dmodule_wrapper_2/batch_normalization_2/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������f�
Imodule_wrapper_2/batch_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
7module_wrapper_2/batch_normalization_2/moments/varianceMeanDmodule_wrapper_2/batch_normalization_2/moments/SquaredDifference:z:0Rmodule_wrapper_2/batch_normalization_2/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:f*
	keep_dims(�
6module_wrapper_2/batch_normalization_2/moments/SqueezeSqueeze<module_wrapper_2/batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes
:f*
squeeze_dims
 �
8module_wrapper_2/batch_normalization_2/moments/Squeeze_1Squeeze@module_wrapper_2/batch_normalization_2/moments/variance:output:0*
T0*
_output_shapes
:f*
squeeze_dims
 �
<module_wrapper_2/batch_normalization_2/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
Emodule_wrapper_2/batch_normalization_2/AssignMovingAvg/ReadVariableOpReadVariableOpNmodule_wrapper_2_batch_normalization_2_assignmovingavg_readvariableop_resource*
_output_shapes
:f*
dtype0�
:module_wrapper_2/batch_normalization_2/AssignMovingAvg/subSubMmodule_wrapper_2/batch_normalization_2/AssignMovingAvg/ReadVariableOp:value:0?module_wrapper_2/batch_normalization_2/moments/Squeeze:output:0*
T0*
_output_shapes
:f�
:module_wrapper_2/batch_normalization_2/AssignMovingAvg/mulMul>module_wrapper_2/batch_normalization_2/AssignMovingAvg/sub:z:0Emodule_wrapper_2/batch_normalization_2/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:f�
6module_wrapper_2/batch_normalization_2/AssignMovingAvgAssignSubVariableOpNmodule_wrapper_2_batch_normalization_2_assignmovingavg_readvariableop_resource>module_wrapper_2/batch_normalization_2/AssignMovingAvg/mul:z:0F^module_wrapper_2/batch_normalization_2/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0�
>module_wrapper_2/batch_normalization_2/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
Gmodule_wrapper_2/batch_normalization_2/AssignMovingAvg_1/ReadVariableOpReadVariableOpPmodule_wrapper_2_batch_normalization_2_assignmovingavg_1_readvariableop_resource*
_output_shapes
:f*
dtype0�
<module_wrapper_2/batch_normalization_2/AssignMovingAvg_1/subSubOmodule_wrapper_2/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp:value:0Amodule_wrapper_2/batch_normalization_2/moments/Squeeze_1:output:0*
T0*
_output_shapes
:f�
<module_wrapper_2/batch_normalization_2/AssignMovingAvg_1/mulMul@module_wrapper_2/batch_normalization_2/AssignMovingAvg_1/sub:z:0Gmodule_wrapper_2/batch_normalization_2/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:f�
8module_wrapper_2/batch_normalization_2/AssignMovingAvg_1AssignSubVariableOpPmodule_wrapper_2_batch_normalization_2_assignmovingavg_1_readvariableop_resource@module_wrapper_2/batch_normalization_2/AssignMovingAvg_1/mul:z:0H^module_wrapper_2/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0{
6module_wrapper_2/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
4module_wrapper_2/batch_normalization_2/batchnorm/addAddV2Amodule_wrapper_2/batch_normalization_2/moments/Squeeze_1:output:0?module_wrapper_2/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:f�
6module_wrapper_2/batch_normalization_2/batchnorm/RsqrtRsqrt8module_wrapper_2/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:f�
Cmodule_wrapper_2/batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpLmodule_wrapper_2_batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:f*
dtype0�
4module_wrapper_2/batch_normalization_2/batchnorm/mulMul:module_wrapper_2/batch_normalization_2/batchnorm/Rsqrt:y:0Kmodule_wrapper_2/batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:f�
6module_wrapper_2/batch_normalization_2/batchnorm/mul_1Muldense_10/Relu:activations:08module_wrapper_2/batch_normalization_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������f�
6module_wrapper_2/batch_normalization_2/batchnorm/mul_2Mul?module_wrapper_2/batch_normalization_2/moments/Squeeze:output:08module_wrapper_2/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:f�
?module_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOpReadVariableOpHmodule_wrapper_2_batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:f*
dtype0�
4module_wrapper_2/batch_normalization_2/batchnorm/subSubGmodule_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp:value:0:module_wrapper_2/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:f�
6module_wrapper_2/batch_normalization_2/batchnorm/add_1AddV2:module_wrapper_2/batch_normalization_2/batchnorm/mul_1:z:08module_wrapper_2/batch_normalization_2/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������f�
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes
:	f�*
dtype0�
dense_11/MatMulMatMul:module_wrapper_2/batch_normalization_2/batchnorm/add_1:z:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������i
IdentityIdentitydense_11/BiasAdd:output:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp7^module_wrapper_2/batch_normalization_2/AssignMovingAvgF^module_wrapper_2/batch_normalization_2/AssignMovingAvg/ReadVariableOp9^module_wrapper_2/batch_normalization_2/AssignMovingAvg_1H^module_wrapper_2/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp@^module_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOpD^module_wrapper_2/batch_normalization_2/batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������R: : : : : : : : : : : : 2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp2p
6module_wrapper_2/batch_normalization_2/AssignMovingAvg6module_wrapper_2/batch_normalization_2/AssignMovingAvg2�
Emodule_wrapper_2/batch_normalization_2/AssignMovingAvg/ReadVariableOpEmodule_wrapper_2/batch_normalization_2/AssignMovingAvg/ReadVariableOp2t
8module_wrapper_2/batch_normalization_2/AssignMovingAvg_18module_wrapper_2/batch_normalization_2/AssignMovingAvg_12�
Gmodule_wrapper_2/batch_normalization_2/AssignMovingAvg_1/ReadVariableOpGmodule_wrapper_2/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp2�
?module_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp?module_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp2�
Cmodule_wrapper_2/batch_normalization_2/batchnorm/mul/ReadVariableOpCmodule_wrapper_2/batch_normalization_2/batchnorm/mul/ReadVariableOp:V R
'
_output_shapes
:���������R
'
_user_specified_namedense_8_input
�	
�
,__inference_dense_11_layer_call_fn_163861864

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
�Q
�
$__inference__wrapped_model_163860745
dense_8_inputE
3sequential_2_dense_8_matmul_readvariableop_resource:RfB
4sequential_2_dense_8_biasadd_readvariableop_resource:fE
3sequential_2_dense_9_matmul_readvariableop_resource:ffB
4sequential_2_dense_9_biasadd_readvariableop_resource:fF
4sequential_2_dense_10_matmul_readvariableop_resource:ffC
5sequential_2_dense_10_biasadd_readvariableop_resource:fc
Usequential_2_module_wrapper_2_batch_normalization_2_batchnorm_readvariableop_resource:fg
Ysequential_2_module_wrapper_2_batch_normalization_2_batchnorm_mul_readvariableop_resource:fe
Wsequential_2_module_wrapper_2_batch_normalization_2_batchnorm_readvariableop_1_resource:fe
Wsequential_2_module_wrapper_2_batch_normalization_2_batchnorm_readvariableop_2_resource:fG
4sequential_2_dense_11_matmul_readvariableop_resource:	f�D
5sequential_2_dense_11_biasadd_readvariableop_resource:	�
identity��,sequential_2/dense_10/BiasAdd/ReadVariableOp�+sequential_2/dense_10/MatMul/ReadVariableOp�,sequential_2/dense_11/BiasAdd/ReadVariableOp�+sequential_2/dense_11/MatMul/ReadVariableOp�+sequential_2/dense_8/BiasAdd/ReadVariableOp�*sequential_2/dense_8/MatMul/ReadVariableOp�+sequential_2/dense_9/BiasAdd/ReadVariableOp�*sequential_2/dense_9/MatMul/ReadVariableOp�Lsequential_2/module_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp�Nsequential_2/module_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp_1�Nsequential_2/module_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp_2�Psequential_2/module_wrapper_2/batch_normalization_2/batchnorm/mul/ReadVariableOp�
*sequential_2/dense_8/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_8_matmul_readvariableop_resource*
_output_shapes

:Rf*
dtype0�
sequential_2/dense_8/MatMulMatMuldense_8_input2sequential_2/dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f�
+sequential_2/dense_8/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_8_biasadd_readvariableop_resource*
_output_shapes
:f*
dtype0�
sequential_2/dense_8/BiasAddBiasAdd%sequential_2/dense_8/MatMul:product:03sequential_2/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������fz
sequential_2/dense_8/ReluRelu%sequential_2/dense_8/BiasAdd:output:0*
T0*'
_output_shapes
:���������f�
*sequential_2/dense_9/MatMul/ReadVariableOpReadVariableOp3sequential_2_dense_9_matmul_readvariableop_resource*
_output_shapes

:ff*
dtype0�
sequential_2/dense_9/MatMulMatMul'sequential_2/dense_8/Relu:activations:02sequential_2/dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f�
+sequential_2/dense_9/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_dense_9_biasadd_readvariableop_resource*
_output_shapes
:f*
dtype0�
sequential_2/dense_9/BiasAddBiasAdd%sequential_2/dense_9/MatMul:product:03sequential_2/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������fz
sequential_2/dense_9/ReluRelu%sequential_2/dense_9/BiasAdd:output:0*
T0*'
_output_shapes
:���������f�
+sequential_2/dense_10/MatMul/ReadVariableOpReadVariableOp4sequential_2_dense_10_matmul_readvariableop_resource*
_output_shapes

:ff*
dtype0�
sequential_2/dense_10/MatMulMatMul'sequential_2/dense_9/Relu:activations:03sequential_2/dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f�
,sequential_2/dense_10/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_dense_10_biasadd_readvariableop_resource*
_output_shapes
:f*
dtype0�
sequential_2/dense_10/BiasAddBiasAdd&sequential_2/dense_10/MatMul:product:04sequential_2/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f|
sequential_2/dense_10/ReluRelu&sequential_2/dense_10/BiasAdd:output:0*
T0*'
_output_shapes
:���������f�
Lsequential_2/module_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOpReadVariableOpUsequential_2_module_wrapper_2_batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:f*
dtype0�
Csequential_2/module_wrapper_2/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
Asequential_2/module_wrapper_2/batch_normalization_2/batchnorm/addAddV2Tsequential_2/module_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp:value:0Lsequential_2/module_wrapper_2/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:f�
Csequential_2/module_wrapper_2/batch_normalization_2/batchnorm/RsqrtRsqrtEsequential_2/module_wrapper_2/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:f�
Psequential_2/module_wrapper_2/batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpYsequential_2_module_wrapper_2_batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:f*
dtype0�
Asequential_2/module_wrapper_2/batch_normalization_2/batchnorm/mulMulGsequential_2/module_wrapper_2/batch_normalization_2/batchnorm/Rsqrt:y:0Xsequential_2/module_wrapper_2/batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:f�
Csequential_2/module_wrapper_2/batch_normalization_2/batchnorm/mul_1Mul(sequential_2/dense_10/Relu:activations:0Esequential_2/module_wrapper_2/batch_normalization_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������f�
Nsequential_2/module_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOpWsequential_2_module_wrapper_2_batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes
:f*
dtype0�
Csequential_2/module_wrapper_2/batch_normalization_2/batchnorm/mul_2MulVsequential_2/module_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp_1:value:0Esequential_2/module_wrapper_2/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:f�
Nsequential_2/module_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOpWsequential_2_module_wrapper_2_batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes
:f*
dtype0�
Asequential_2/module_wrapper_2/batch_normalization_2/batchnorm/subSubVsequential_2/module_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp_2:value:0Gsequential_2/module_wrapper_2/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:f�
Csequential_2/module_wrapper_2/batch_normalization_2/batchnorm/add_1AddV2Gsequential_2/module_wrapper_2/batch_normalization_2/batchnorm/mul_1:z:0Esequential_2/module_wrapper_2/batch_normalization_2/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������f�
sequential_2/dropout_2/IdentityIdentityGsequential_2/module_wrapper_2/batch_normalization_2/batchnorm/add_1:z:0*
T0*'
_output_shapes
:���������f�
+sequential_2/dense_11/MatMul/ReadVariableOpReadVariableOp4sequential_2_dense_11_matmul_readvariableop_resource*
_output_shapes
:	f�*
dtype0�
sequential_2/dense_11/MatMulMatMul(sequential_2/dropout_2/Identity:output:03sequential_2/dense_11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
,sequential_2/dense_11/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_dense_11_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential_2/dense_11/BiasAddBiasAdd&sequential_2/dense_11/MatMul:product:04sequential_2/dense_11/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������v
IdentityIdentity&sequential_2/dense_11/BiasAdd:output:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp-^sequential_2/dense_10/BiasAdd/ReadVariableOp,^sequential_2/dense_10/MatMul/ReadVariableOp-^sequential_2/dense_11/BiasAdd/ReadVariableOp,^sequential_2/dense_11/MatMul/ReadVariableOp,^sequential_2/dense_8/BiasAdd/ReadVariableOp+^sequential_2/dense_8/MatMul/ReadVariableOp,^sequential_2/dense_9/BiasAdd/ReadVariableOp+^sequential_2/dense_9/MatMul/ReadVariableOpM^sequential_2/module_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOpO^sequential_2/module_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp_1O^sequential_2/module_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp_2Q^sequential_2/module_wrapper_2/batch_normalization_2/batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������R: : : : : : : : : : : : 2\
,sequential_2/dense_10/BiasAdd/ReadVariableOp,sequential_2/dense_10/BiasAdd/ReadVariableOp2Z
+sequential_2/dense_10/MatMul/ReadVariableOp+sequential_2/dense_10/MatMul/ReadVariableOp2\
,sequential_2/dense_11/BiasAdd/ReadVariableOp,sequential_2/dense_11/BiasAdd/ReadVariableOp2Z
+sequential_2/dense_11/MatMul/ReadVariableOp+sequential_2/dense_11/MatMul/ReadVariableOp2Z
+sequential_2/dense_8/BiasAdd/ReadVariableOp+sequential_2/dense_8/BiasAdd/ReadVariableOp2X
*sequential_2/dense_8/MatMul/ReadVariableOp*sequential_2/dense_8/MatMul/ReadVariableOp2Z
+sequential_2/dense_9/BiasAdd/ReadVariableOp+sequential_2/dense_9/BiasAdd/ReadVariableOp2X
*sequential_2/dense_9/MatMul/ReadVariableOp*sequential_2/dense_9/MatMul/ReadVariableOp2�
Lsequential_2/module_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOpLsequential_2/module_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp2�
Nsequential_2/module_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp_1Nsequential_2/module_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp_12�
Nsequential_2/module_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp_2Nsequential_2/module_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp_22�
Psequential_2/module_wrapper_2/batch_normalization_2/batchnorm/mul/ReadVariableOpPsequential_2/module_wrapper_2/batch_normalization_2/batchnorm/mul/ReadVariableOp:V R
'
_output_shapes
:���������R
'
_user_specified_namedense_8_input
�
K
-__inference_dropout_2_layer_call_fn_163861841

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
�
f
H__inference_dropout_2_layer_call_and_return_conditional_losses_163861850

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
�5
�
4__inference_module_wrapper_2_layer_call_fn_163861782

args_0K
=batch_normalization_2_assignmovingavg_readvariableop_resource:fM
?batch_normalization_2_assignmovingavg_1_readvariableop_resource:fI
;batch_normalization_2_batchnorm_mul_readvariableop_resource:fE
7batch_normalization_2_batchnorm_readvariableop_resource:f
identity��%batch_normalization_2/AssignMovingAvg�4batch_normalization_2/AssignMovingAvg/ReadVariableOp�'batch_normalization_2/AssignMovingAvg_1�6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp�.batch_normalization_2/batchnorm/ReadVariableOp�2batch_normalization_2/batchnorm/mul/ReadVariableOp~
4batch_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
"batch_normalization_2/moments/meanMeanargs_0=batch_normalization_2/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:f*
	keep_dims(�
*batch_normalization_2/moments/StopGradientStopGradient+batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes

:f�
/batch_normalization_2/moments/SquaredDifferenceSquaredDifferenceargs_03batch_normalization_2/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������f�
8batch_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
&batch_normalization_2/moments/varianceMean3batch_normalization_2/moments/SquaredDifference:z:0Abatch_normalization_2/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:f*
	keep_dims(�
%batch_normalization_2/moments/SqueezeSqueeze+batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes
:f*
squeeze_dims
 �
'batch_normalization_2/moments/Squeeze_1Squeeze/batch_normalization_2/moments/variance:output:0*
T0*
_output_shapes
:f*
squeeze_dims
 p
+batch_normalization_2/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
4batch_normalization_2/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_2_assignmovingavg_readvariableop_resource*
_output_shapes
:f*
dtype0�
)batch_normalization_2/AssignMovingAvg/subSub<batch_normalization_2/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_2/moments/Squeeze:output:0*
T0*
_output_shapes
:f�
)batch_normalization_2/AssignMovingAvg/mulMul-batch_normalization_2/AssignMovingAvg/sub:z:04batch_normalization_2/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:f�
%batch_normalization_2/AssignMovingAvgAssignSubVariableOp=batch_normalization_2_assignmovingavg_readvariableop_resource-batch_normalization_2/AssignMovingAvg/mul:z:05^batch_normalization_2/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0r
-batch_normalization_2/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_2_assignmovingavg_1_readvariableop_resource*
_output_shapes
:f*
dtype0�
+batch_normalization_2/AssignMovingAvg_1/subSub>batch_normalization_2/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_2/moments/Squeeze_1:output:0*
T0*
_output_shapes
:f�
+batch_normalization_2/AssignMovingAvg_1/mulMul/batch_normalization_2/AssignMovingAvg_1/sub:z:06batch_normalization_2/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:f�
'batch_normalization_2/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_2_assignmovingavg_1_readvariableop_resource/batch_normalization_2/AssignMovingAvg_1/mul:z:07^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0j
%batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
#batch_normalization_2/batchnorm/addAddV20batch_normalization_2/moments/Squeeze_1:output:0.batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:f|
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:f�
2batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:f*
dtype0�
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:0:batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:f�
%batch_normalization_2/batchnorm/mul_1Mulargs_0'batch_normalization_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������f�
%batch_normalization_2/batchnorm/mul_2Mul.batch_normalization_2/moments/Squeeze:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:f�
.batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:f*
dtype0�
#batch_normalization_2/batchnorm/subSub6batch_normalization_2/batchnorm/ReadVariableOp:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:f�
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������fx
IdentityIdentity)batch_normalization_2/batchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������f�
NoOpNoOp&^batch_normalization_2/AssignMovingAvg5^batch_normalization_2/AssignMovingAvg/ReadVariableOp(^batch_normalization_2/AssignMovingAvg_17^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_2/batchnorm/ReadVariableOp3^batch_normalization_2/batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������f: : : : 2N
%batch_normalization_2/AssignMovingAvg%batch_normalization_2/AssignMovingAvg2l
4batch_normalization_2/AssignMovingAvg/ReadVariableOp4batch_normalization_2/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_2/AssignMovingAvg_1'batch_normalization_2/AssignMovingAvg_12p
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_2/batchnorm/ReadVariableOp.batch_normalization_2/batchnorm/ReadVariableOp2h
2batch_normalization_2/batchnorm/mul/ReadVariableOp2batch_normalization_2/batchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������f
 
_user_specified_nameargs_0
�
�
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_163862059

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
+__inference_dense_8_layer_call_fn_163861673

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
K__inference_sequential_2_layer_call_and_return_conditional_losses_163861407
dense_8_input8
&dense_8_matmul_readvariableop_resource:Rf5
'dense_8_biasadd_readvariableop_resource:f8
&dense_9_matmul_readvariableop_resource:ff5
'dense_9_biasadd_readvariableop_resource:f9
'dense_10_matmul_readvariableop_resource:ff6
(dense_10_biasadd_readvariableop_resource:f\
Nmodule_wrapper_2_batch_normalization_2_assignmovingavg_readvariableop_resource:f^
Pmodule_wrapper_2_batch_normalization_2_assignmovingavg_1_readvariableop_resource:fZ
Lmodule_wrapper_2_batch_normalization_2_batchnorm_mul_readvariableop_resource:fV
Hmodule_wrapper_2_batch_normalization_2_batchnorm_readvariableop_resource:f:
'dense_11_matmul_readvariableop_resource:	f�7
(dense_11_biasadd_readvariableop_resource:	�
identity��dense_10/BiasAdd/ReadVariableOp�dense_10/MatMul/ReadVariableOp�dense_11/BiasAdd/ReadVariableOp�dense_11/MatMul/ReadVariableOp�dense_8/BiasAdd/ReadVariableOp�dense_8/MatMul/ReadVariableOp�dense_9/BiasAdd/ReadVariableOp�dense_9/MatMul/ReadVariableOp�6module_wrapper_2/batch_normalization_2/AssignMovingAvg�Emodule_wrapper_2/batch_normalization_2/AssignMovingAvg/ReadVariableOp�8module_wrapper_2/batch_normalization_2/AssignMovingAvg_1�Gmodule_wrapper_2/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp�?module_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp�Cmodule_wrapper_2/batch_normalization_2/batchnorm/mul/ReadVariableOp�
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes

:Rf*
dtype0�
dense_8/MatMulMatMuldense_8_input%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f�
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:f*
dtype0�
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f`
dense_8/ReluReludense_8/BiasAdd:output:0*
T0*'
_output_shapes
:���������f�
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes

:ff*
dtype0�
dense_9/MatMulMatMuldense_8/Relu:activations:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f�
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:f*
dtype0�
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f`
dense_9/ReluReludense_9/BiasAdd:output:0*
T0*'
_output_shapes
:���������f�
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource*
_output_shapes

:ff*
dtype0�
dense_10/MatMulMatMuldense_9/Relu:activations:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f�
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes
:f*
dtype0�
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������fb
dense_10/ReluReludense_10/BiasAdd:output:0*
T0*'
_output_shapes
:���������f�
Emodule_wrapper_2/batch_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
3module_wrapper_2/batch_normalization_2/moments/meanMeandense_10/Relu:activations:0Nmodule_wrapper_2/batch_normalization_2/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:f*
	keep_dims(�
;module_wrapper_2/batch_normalization_2/moments/StopGradientStopGradient<module_wrapper_2/batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes

:f�
@module_wrapper_2/batch_normalization_2/moments/SquaredDifferenceSquaredDifferencedense_10/Relu:activations:0Dmodule_wrapper_2/batch_normalization_2/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������f�
Imodule_wrapper_2/batch_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
7module_wrapper_2/batch_normalization_2/moments/varianceMeanDmodule_wrapper_2/batch_normalization_2/moments/SquaredDifference:z:0Rmodule_wrapper_2/batch_normalization_2/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:f*
	keep_dims(�
6module_wrapper_2/batch_normalization_2/moments/SqueezeSqueeze<module_wrapper_2/batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes
:f*
squeeze_dims
 �
8module_wrapper_2/batch_normalization_2/moments/Squeeze_1Squeeze@module_wrapper_2/batch_normalization_2/moments/variance:output:0*
T0*
_output_shapes
:f*
squeeze_dims
 �
<module_wrapper_2/batch_normalization_2/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
Emodule_wrapper_2/batch_normalization_2/AssignMovingAvg/ReadVariableOpReadVariableOpNmodule_wrapper_2_batch_normalization_2_assignmovingavg_readvariableop_resource*
_output_shapes
:f*
dtype0�
:module_wrapper_2/batch_normalization_2/AssignMovingAvg/subSubMmodule_wrapper_2/batch_normalization_2/AssignMovingAvg/ReadVariableOp:value:0?module_wrapper_2/batch_normalization_2/moments/Squeeze:output:0*
T0*
_output_shapes
:f�
:module_wrapper_2/batch_normalization_2/AssignMovingAvg/mulMul>module_wrapper_2/batch_normalization_2/AssignMovingAvg/sub:z:0Emodule_wrapper_2/batch_normalization_2/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:f�
6module_wrapper_2/batch_normalization_2/AssignMovingAvgAssignSubVariableOpNmodule_wrapper_2_batch_normalization_2_assignmovingavg_readvariableop_resource>module_wrapper_2/batch_normalization_2/AssignMovingAvg/mul:z:0F^module_wrapper_2/batch_normalization_2/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0�
>module_wrapper_2/batch_normalization_2/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
Gmodule_wrapper_2/batch_normalization_2/AssignMovingAvg_1/ReadVariableOpReadVariableOpPmodule_wrapper_2_batch_normalization_2_assignmovingavg_1_readvariableop_resource*
_output_shapes
:f*
dtype0�
<module_wrapper_2/batch_normalization_2/AssignMovingAvg_1/subSubOmodule_wrapper_2/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp:value:0Amodule_wrapper_2/batch_normalization_2/moments/Squeeze_1:output:0*
T0*
_output_shapes
:f�
<module_wrapper_2/batch_normalization_2/AssignMovingAvg_1/mulMul@module_wrapper_2/batch_normalization_2/AssignMovingAvg_1/sub:z:0Gmodule_wrapper_2/batch_normalization_2/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:f�
8module_wrapper_2/batch_normalization_2/AssignMovingAvg_1AssignSubVariableOpPmodule_wrapper_2_batch_normalization_2_assignmovingavg_1_readvariableop_resource@module_wrapper_2/batch_normalization_2/AssignMovingAvg_1/mul:z:0H^module_wrapper_2/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0{
6module_wrapper_2/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
4module_wrapper_2/batch_normalization_2/batchnorm/addAddV2Amodule_wrapper_2/batch_normalization_2/moments/Squeeze_1:output:0?module_wrapper_2/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:f�
6module_wrapper_2/batch_normalization_2/batchnorm/RsqrtRsqrt8module_wrapper_2/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:f�
Cmodule_wrapper_2/batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpLmodule_wrapper_2_batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:f*
dtype0�
4module_wrapper_2/batch_normalization_2/batchnorm/mulMul:module_wrapper_2/batch_normalization_2/batchnorm/Rsqrt:y:0Kmodule_wrapper_2/batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:f�
6module_wrapper_2/batch_normalization_2/batchnorm/mul_1Muldense_10/Relu:activations:08module_wrapper_2/batch_normalization_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������f�
6module_wrapper_2/batch_normalization_2/batchnorm/mul_2Mul?module_wrapper_2/batch_normalization_2/moments/Squeeze:output:08module_wrapper_2/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:f�
?module_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOpReadVariableOpHmodule_wrapper_2_batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:f*
dtype0�
4module_wrapper_2/batch_normalization_2/batchnorm/subSubGmodule_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp:value:0:module_wrapper_2/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:f�
6module_wrapper_2/batch_normalization_2/batchnorm/add_1AddV2:module_wrapper_2/batch_normalization_2/batchnorm/mul_1:z:08module_wrapper_2/batch_normalization_2/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������f�
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes
:	f�*
dtype0�
dense_11/MatMulMatMul:module_wrapper_2/batch_normalization_2/batchnorm/add_1:z:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������i
IdentityIdentitydense_11/BiasAdd:output:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp7^module_wrapper_2/batch_normalization_2/AssignMovingAvgF^module_wrapper_2/batch_normalization_2/AssignMovingAvg/ReadVariableOp9^module_wrapper_2/batch_normalization_2/AssignMovingAvg_1H^module_wrapper_2/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp@^module_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOpD^module_wrapper_2/batch_normalization_2/batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������R: : : : : : : : : : : : 2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp2p
6module_wrapper_2/batch_normalization_2/AssignMovingAvg6module_wrapper_2/batch_normalization_2/AssignMovingAvg2�
Emodule_wrapper_2/batch_normalization_2/AssignMovingAvg/ReadVariableOpEmodule_wrapper_2/batch_normalization_2/AssignMovingAvg/ReadVariableOp2t
8module_wrapper_2/batch_normalization_2/AssignMovingAvg_18module_wrapper_2/batch_normalization_2/AssignMovingAvg_12�
Gmodule_wrapper_2/batch_normalization_2/AssignMovingAvg_1/ReadVariableOpGmodule_wrapper_2/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp2�
?module_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp?module_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp2�
Cmodule_wrapper_2/batch_normalization_2/batchnorm/mul/ReadVariableOpCmodule_wrapper_2/batch_normalization_2/batchnorm/mul/ReadVariableOp:V R
'
_output_shapes
:���������R
'
_user_specified_namedense_8_input
�a
�
0__inference_sequential_2_layer_call_fn_163861522

inputs8
&dense_8_matmul_readvariableop_resource:Rf5
'dense_8_biasadd_readvariableop_resource:f8
&dense_9_matmul_readvariableop_resource:ff5
'dense_9_biasadd_readvariableop_resource:f9
'dense_10_matmul_readvariableop_resource:ff6
(dense_10_biasadd_readvariableop_resource:f\
Nmodule_wrapper_2_batch_normalization_2_assignmovingavg_readvariableop_resource:f^
Pmodule_wrapper_2_batch_normalization_2_assignmovingavg_1_readvariableop_resource:fZ
Lmodule_wrapper_2_batch_normalization_2_batchnorm_mul_readvariableop_resource:fV
Hmodule_wrapper_2_batch_normalization_2_batchnorm_readvariableop_resource:f:
'dense_11_matmul_readvariableop_resource:	f�7
(dense_11_biasadd_readvariableop_resource:	�
identity��dense_10/BiasAdd/ReadVariableOp�dense_10/MatMul/ReadVariableOp�dense_11/BiasAdd/ReadVariableOp�dense_11/MatMul/ReadVariableOp�dense_8/BiasAdd/ReadVariableOp�dense_8/MatMul/ReadVariableOp�dense_9/BiasAdd/ReadVariableOp�dense_9/MatMul/ReadVariableOp�6module_wrapper_2/batch_normalization_2/AssignMovingAvg�Emodule_wrapper_2/batch_normalization_2/AssignMovingAvg/ReadVariableOp�8module_wrapper_2/batch_normalization_2/AssignMovingAvg_1�Gmodule_wrapper_2/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp�?module_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp�Cmodule_wrapper_2/batch_normalization_2/batchnorm/mul/ReadVariableOp�
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes

:Rf*
dtype0y
dense_8/MatMulMatMulinputs%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f�
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:f*
dtype0�
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f`
dense_8/ReluReludense_8/BiasAdd:output:0*
T0*'
_output_shapes
:���������f�
dense_9/MatMul/ReadVariableOpReadVariableOp&dense_9_matmul_readvariableop_resource*
_output_shapes

:ff*
dtype0�
dense_9/MatMulMatMuldense_8/Relu:activations:0%dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f�
dense_9/BiasAdd/ReadVariableOpReadVariableOp'dense_9_biasadd_readvariableop_resource*
_output_shapes
:f*
dtype0�
dense_9/BiasAddBiasAdddense_9/MatMul:product:0&dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f`
dense_9/ReluReludense_9/BiasAdd:output:0*
T0*'
_output_shapes
:���������f�
dense_10/MatMul/ReadVariableOpReadVariableOp'dense_10_matmul_readvariableop_resource*
_output_shapes

:ff*
dtype0�
dense_10/MatMulMatMuldense_9/Relu:activations:0&dense_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������f�
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes
:f*
dtype0�
dense_10/BiasAddBiasAdddense_10/MatMul:product:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������fb
dense_10/ReluReludense_10/BiasAdd:output:0*
T0*'
_output_shapes
:���������f�
Emodule_wrapper_2/batch_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
3module_wrapper_2/batch_normalization_2/moments/meanMeandense_10/Relu:activations:0Nmodule_wrapper_2/batch_normalization_2/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:f*
	keep_dims(�
;module_wrapper_2/batch_normalization_2/moments/StopGradientStopGradient<module_wrapper_2/batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes

:f�
@module_wrapper_2/batch_normalization_2/moments/SquaredDifferenceSquaredDifferencedense_10/Relu:activations:0Dmodule_wrapper_2/batch_normalization_2/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������f�
Imodule_wrapper_2/batch_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
7module_wrapper_2/batch_normalization_2/moments/varianceMeanDmodule_wrapper_2/batch_normalization_2/moments/SquaredDifference:z:0Rmodule_wrapper_2/batch_normalization_2/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:f*
	keep_dims(�
6module_wrapper_2/batch_normalization_2/moments/SqueezeSqueeze<module_wrapper_2/batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes
:f*
squeeze_dims
 �
8module_wrapper_2/batch_normalization_2/moments/Squeeze_1Squeeze@module_wrapper_2/batch_normalization_2/moments/variance:output:0*
T0*
_output_shapes
:f*
squeeze_dims
 �
<module_wrapper_2/batch_normalization_2/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
Emodule_wrapper_2/batch_normalization_2/AssignMovingAvg/ReadVariableOpReadVariableOpNmodule_wrapper_2_batch_normalization_2_assignmovingavg_readvariableop_resource*
_output_shapes
:f*
dtype0�
:module_wrapper_2/batch_normalization_2/AssignMovingAvg/subSubMmodule_wrapper_2/batch_normalization_2/AssignMovingAvg/ReadVariableOp:value:0?module_wrapper_2/batch_normalization_2/moments/Squeeze:output:0*
T0*
_output_shapes
:f�
:module_wrapper_2/batch_normalization_2/AssignMovingAvg/mulMul>module_wrapper_2/batch_normalization_2/AssignMovingAvg/sub:z:0Emodule_wrapper_2/batch_normalization_2/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:f�
6module_wrapper_2/batch_normalization_2/AssignMovingAvgAssignSubVariableOpNmodule_wrapper_2_batch_normalization_2_assignmovingavg_readvariableop_resource>module_wrapper_2/batch_normalization_2/AssignMovingAvg/mul:z:0F^module_wrapper_2/batch_normalization_2/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0�
>module_wrapper_2/batch_normalization_2/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
Gmodule_wrapper_2/batch_normalization_2/AssignMovingAvg_1/ReadVariableOpReadVariableOpPmodule_wrapper_2_batch_normalization_2_assignmovingavg_1_readvariableop_resource*
_output_shapes
:f*
dtype0�
<module_wrapper_2/batch_normalization_2/AssignMovingAvg_1/subSubOmodule_wrapper_2/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp:value:0Amodule_wrapper_2/batch_normalization_2/moments/Squeeze_1:output:0*
T0*
_output_shapes
:f�
<module_wrapper_2/batch_normalization_2/AssignMovingAvg_1/mulMul@module_wrapper_2/batch_normalization_2/AssignMovingAvg_1/sub:z:0Gmodule_wrapper_2/batch_normalization_2/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:f�
8module_wrapper_2/batch_normalization_2/AssignMovingAvg_1AssignSubVariableOpPmodule_wrapper_2_batch_normalization_2_assignmovingavg_1_readvariableop_resource@module_wrapper_2/batch_normalization_2/AssignMovingAvg_1/mul:z:0H^module_wrapper_2/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0{
6module_wrapper_2/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
4module_wrapper_2/batch_normalization_2/batchnorm/addAddV2Amodule_wrapper_2/batch_normalization_2/moments/Squeeze_1:output:0?module_wrapper_2/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:f�
6module_wrapper_2/batch_normalization_2/batchnorm/RsqrtRsqrt8module_wrapper_2/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:f�
Cmodule_wrapper_2/batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpLmodule_wrapper_2_batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:f*
dtype0�
4module_wrapper_2/batch_normalization_2/batchnorm/mulMul:module_wrapper_2/batch_normalization_2/batchnorm/Rsqrt:y:0Kmodule_wrapper_2/batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:f�
6module_wrapper_2/batch_normalization_2/batchnorm/mul_1Muldense_10/Relu:activations:08module_wrapper_2/batch_normalization_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������f�
6module_wrapper_2/batch_normalization_2/batchnorm/mul_2Mul?module_wrapper_2/batch_normalization_2/moments/Squeeze:output:08module_wrapper_2/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:f�
?module_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOpReadVariableOpHmodule_wrapper_2_batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:f*
dtype0�
4module_wrapper_2/batch_normalization_2/batchnorm/subSubGmodule_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp:value:0:module_wrapper_2/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:f�
6module_wrapper_2/batch_normalization_2/batchnorm/add_1AddV2:module_wrapper_2/batch_normalization_2/batchnorm/mul_1:z:08module_wrapper_2/batch_normalization_2/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������f�
dense_11/MatMul/ReadVariableOpReadVariableOp'dense_11_matmul_readvariableop_resource*
_output_shapes
:	f�*
dtype0�
dense_11/MatMulMatMul:module_wrapper_2/batch_normalization_2/batchnorm/add_1:z:0&dense_11/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_11/BiasAdd/ReadVariableOpReadVariableOp(dense_11_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_11/BiasAddBiasAdddense_11/MatMul:product:0'dense_11/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������i
IdentityIdentitydense_11/BiasAdd:output:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp ^dense_10/BiasAdd/ReadVariableOp^dense_10/MatMul/ReadVariableOp ^dense_11/BiasAdd/ReadVariableOp^dense_11/MatMul/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp^dense_9/BiasAdd/ReadVariableOp^dense_9/MatMul/ReadVariableOp7^module_wrapper_2/batch_normalization_2/AssignMovingAvgF^module_wrapper_2/batch_normalization_2/AssignMovingAvg/ReadVariableOp9^module_wrapper_2/batch_normalization_2/AssignMovingAvg_1H^module_wrapper_2/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp@^module_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOpD^module_wrapper_2/batch_normalization_2/batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:���������R: : : : : : : : : : : : 2B
dense_10/BiasAdd/ReadVariableOpdense_10/BiasAdd/ReadVariableOp2@
dense_10/MatMul/ReadVariableOpdense_10/MatMul/ReadVariableOp2B
dense_11/BiasAdd/ReadVariableOpdense_11/BiasAdd/ReadVariableOp2@
dense_11/MatMul/ReadVariableOpdense_11/MatMul/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp2@
dense_9/BiasAdd/ReadVariableOpdense_9/BiasAdd/ReadVariableOp2>
dense_9/MatMul/ReadVariableOpdense_9/MatMul/ReadVariableOp2p
6module_wrapper_2/batch_normalization_2/AssignMovingAvg6module_wrapper_2/batch_normalization_2/AssignMovingAvg2�
Emodule_wrapper_2/batch_normalization_2/AssignMovingAvg/ReadVariableOpEmodule_wrapper_2/batch_normalization_2/AssignMovingAvg/ReadVariableOp2t
8module_wrapper_2/batch_normalization_2/AssignMovingAvg_18module_wrapper_2/batch_normalization_2/AssignMovingAvg_12�
Gmodule_wrapper_2/batch_normalization_2/AssignMovingAvg_1/ReadVariableOpGmodule_wrapper_2/batch_normalization_2/AssignMovingAvg_1/ReadVariableOp2�
?module_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp?module_wrapper_2/batch_normalization_2/batchnorm/ReadVariableOp2�
Cmodule_wrapper_2/batch_normalization_2/batchnorm/mul/ReadVariableOpCmodule_wrapper_2/batch_normalization_2/batchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������R
 
_user_specified_nameinputs
�
�
O__inference_module_wrapper_2_layer_call_and_return_conditional_losses_163861802

args_0E
7batch_normalization_2_batchnorm_readvariableop_resource:fI
;batch_normalization_2_batchnorm_mul_readvariableop_resource:fG
9batch_normalization_2_batchnorm_readvariableop_1_resource:fG
9batch_normalization_2_batchnorm_readvariableop_2_resource:f
identity��.batch_normalization_2/batchnorm/ReadVariableOp�0batch_normalization_2/batchnorm/ReadVariableOp_1�0batch_normalization_2/batchnorm/ReadVariableOp_2�2batch_normalization_2/batchnorm/mul/ReadVariableOp�
.batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:f*
dtype0j
%batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
#batch_normalization_2/batchnorm/addAddV26batch_normalization_2/batchnorm/ReadVariableOp:value:0.batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:f|
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:f�
2batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:f*
dtype0�
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:0:batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:f�
%batch_normalization_2/batchnorm/mul_1Mulargs_0'batch_normalization_2/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������f�
0batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes
:f*
dtype0�
%batch_normalization_2/batchnorm/mul_2Mul8batch_normalization_2/batchnorm/ReadVariableOp_1:value:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:f�
0batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes
:f*
dtype0�
#batch_normalization_2/batchnorm/subSub8batch_normalization_2/batchnorm/ReadVariableOp_2:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:f�
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������fx
IdentityIdentity)batch_normalization_2/batchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������f�
NoOpNoOp/^batch_normalization_2/batchnorm/ReadVariableOp1^batch_normalization_2/batchnorm/ReadVariableOp_11^batch_normalization_2/batchnorm/ReadVariableOp_23^batch_normalization_2/batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������f: : : : 2`
.batch_normalization_2/batchnorm/ReadVariableOp.batch_normalization_2/batchnorm/ReadVariableOp2d
0batch_normalization_2/batchnorm/ReadVariableOp_10batch_normalization_2/batchnorm/ReadVariableOp_12d
0batch_normalization_2/batchnorm/ReadVariableOp_20batch_normalization_2/batchnorm/ReadVariableOp_22h
2batch_normalization_2/batchnorm/mul/ReadVariableOp2batch_normalization_2/batchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������f
 
_user_specified_nameargs_0
�

�
+__inference_dense_9_layer_call_fn_163861695

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
�
I
-__inference_dropout_2_layer_call_fn_163861845

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
�%
�
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_163862093

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
�	
�
G__inference_dense_11_layer_call_and_return_conditional_losses_163861874

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
 
_user_specified_nameinputs"�-
saver_filename:0
Identity:0Identity_448"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
G
dense_8_input6
serving_default_dense_8_input:0���������R=
dense_111
StatefulPartitionedCall:0����������tensorflow/serving/predict:��
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
regularization_losses
	trainable_variables

	variables
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_sequential
�

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
�

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
�

 kernel
!bias
"regularization_losses
#trainable_variables
$	variables
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses"
_tf_keras_layer
�
(_module
)regularization_losses
*trainable_variables
+	variables
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses"
_tf_keras_layer
�
/regularization_losses
0trainable_variables
1	variables
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses"
_tf_keras_layer
�

5kernel
6bias
7regularization_losses
8trainable_variables
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
 "
trackable_list_wrapper
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
regularization_losses
	trainable_variables

Flayers
Gnon_trainable_variables
Hmetrics

	variables
Ilayer_regularization_losses
Jlayer_metrics
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�2�
0__inference_sequential_2_layer_call_fn_163860794
0__inference_sequential_2_layer_call_fn_163861461
0__inference_sequential_2_layer_call_fn_163861522
0__inference_sequential_2_layer_call_fn_163861298�
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
K__inference_sequential_2_layer_call_and_return_conditional_losses_163861570
K__inference_sequential_2_layer_call_and_return_conditional_losses_163861631
K__inference_sequential_2_layer_call_and_return_conditional_losses_163861346
K__inference_sequential_2_layer_call_and_return_conditional_losses_163861407�
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
�2�
$__inference__wrapped_model_163860745�
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
annotations� *,�)
'�$
dense_8_input���������R
,
Kserving_default"
signature_map
 :Rf2dense_8/kernel
:f2dense_8/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
regularization_losses

Llayers
trainable_variables
	variables
Mnon_trainable_variables
Nmetrics
Olayer_regularization_losses
Player_metrics
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�2�
+__inference_dense_8_layer_call_fn_163861673�
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
F__inference_dense_8_layer_call_and_return_conditional_losses_163861684�
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
 :ff2dense_9/kernel
:f2dense_9/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
regularization_losses

Qlayers
trainable_variables
	variables
Rnon_trainable_variables
Smetrics
Tlayer_regularization_losses
Ulayer_metrics
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�2�
+__inference_dense_9_layer_call_fn_163861695�
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
F__inference_dense_9_layer_call_and_return_conditional_losses_163861706�
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
!:ff2dense_10/kernel
:f2dense_10/bias
 "
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
�
"regularization_losses

Vlayers
#trainable_variables
$	variables
Wnon_trainable_variables
Xmetrics
Ylayer_regularization_losses
Zlayer_metrics
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses"
_generic_user_object
�2�
,__inference_dense_10_layer_call_fn_163861717�
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
G__inference_dense_10_layer_call_and_return_conditional_losses_163861728�
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
 "
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
<
B0
C1
D2
E3"
trackable_list_wrapper
�
)regularization_losses

blayers
*trainable_variables
+	variables
cnon_trainable_variables
dmetrics
elayer_regularization_losses
flayer_metrics
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses"
_generic_user_object
�2�
4__inference_module_wrapper_2_layer_call_fn_163861748
4__inference_module_wrapper_2_layer_call_fn_163861782�
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
O__inference_module_wrapper_2_layer_call_and_return_conditional_losses_163861802
O__inference_module_wrapper_2_layer_call_and_return_conditional_losses_163861836�
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
/regularization_losses

glayers
0trainable_variables
1	variables
hnon_trainable_variables
imetrics
jlayer_regularization_losses
klayer_metrics
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses"
_generic_user_object
�2�
-__inference_dropout_2_layer_call_fn_163861841
-__inference_dropout_2_layer_call_fn_163861845�
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
H__inference_dropout_2_layer_call_and_return_conditional_losses_163861850
H__inference_dropout_2_layer_call_and_return_conditional_losses_163861854�
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
": 	f�2dense_11/kernel
:�2dense_11/bias
 "
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
�
7regularization_losses

llayers
8trainable_variables
9	variables
mnon_trainable_variables
nmetrics
olayer_regularization_losses
player_metrics
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses"
_generic_user_object
�2�
,__inference_dense_11_layer_call_fn_163861864�
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
G__inference_dense_11_layer_call_and_return_conditional_losses_163861874�
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
::8f2,module_wrapper_2/batch_normalization_2/gamma
9:7f2+module_wrapper_2/batch_normalization_2/beta
B:@f (22module_wrapper_2/batch_normalization_2/moving_mean
F:Df (26module_wrapper_2/batch_normalization_2/moving_variance
J
0
1
2
3
4
5"
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
.
q0
r1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_signature_wrapper_163861662dense_8_input"�
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
9__inference_batch_normalization_2_layer_call_fn_163862005
9__inference_batch_normalization_2_layer_call_fn_163862039�
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
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_163862059
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_163862093�
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
trackable_list_wrapper
.
D0
E1"
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
%:#Rf2Adam/dense_8/kernel/m
:f2Adam/dense_8/bias/m
%:#ff2Adam/dense_9/kernel/m
:f2Adam/dense_9/bias/m
&:$ff2Adam/dense_10/kernel/m
 :f2Adam/dense_10/bias/m
':%	f�2Adam/dense_11/kernel/m
!:�2Adam/dense_11/bias/m
?:=f23Adam/module_wrapper_2/batch_normalization_2/gamma/m
>:<f22Adam/module_wrapper_2/batch_normalization_2/beta/m
%:#Rf2Adam/dense_8/kernel/v
:f2Adam/dense_8/bias/v
%:#ff2Adam/dense_9/kernel/v
:f2Adam/dense_9/bias/v
&:$ff2Adam/dense_10/kernel/v
 :f2Adam/dense_10/bias/v
':%	f�2Adam/dense_11/kernel/v
!:�2Adam/dense_11/bias/v
?:=f23Adam/module_wrapper_2/batch_normalization_2/gamma/v
>:<f22Adam/module_wrapper_2/batch_normalization_2/beta/v�
$__inference__wrapped_model_163860745| !EBDC566�3
,�)
'�$
dense_8_input���������R
� "4�1
/
dense_11#� 
dense_11�����������
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_163862059bEBDC3�0
)�&
 �
inputs���������f
p 
� "%�"
�
0���������f
� �
T__inference_batch_normalization_2_layer_call_and_return_conditional_losses_163862093bDEBC3�0
)�&
 �
inputs���������f
p
� "%�"
�
0���������f
� �
9__inference_batch_normalization_2_layer_call_fn_163862005UEBDC3�0
)�&
 �
inputs���������f
p 
� "����������f�
9__inference_batch_normalization_2_layer_call_fn_163862039UDEBC3�0
)�&
 �
inputs���������f
p
� "����������f�
G__inference_dense_10_layer_call_and_return_conditional_losses_163861728\ !/�,
%�"
 �
inputs���������f
� "%�"
�
0���������f
� 
,__inference_dense_10_layer_call_fn_163861717O !/�,
%�"
 �
inputs���������f
� "����������f�
G__inference_dense_11_layer_call_and_return_conditional_losses_163861874]56/�,
%�"
 �
inputs���������f
� "&�#
�
0����������
� �
,__inference_dense_11_layer_call_fn_163861864P56/�,
%�"
 �
inputs���������f
� "������������
F__inference_dense_8_layer_call_and_return_conditional_losses_163861684\/�,
%�"
 �
inputs���������R
� "%�"
�
0���������f
� ~
+__inference_dense_8_layer_call_fn_163861673O/�,
%�"
 �
inputs���������R
� "����������f�
F__inference_dense_9_layer_call_and_return_conditional_losses_163861706\/�,
%�"
 �
inputs���������f
� "%�"
�
0���������f
� ~
+__inference_dense_9_layer_call_fn_163861695O/�,
%�"
 �
inputs���������f
� "����������f�
H__inference_dropout_2_layer_call_and_return_conditional_losses_163861850\3�0
)�&
 �
inputs���������f
p 
� "%�"
�
0���������f
� �
H__inference_dropout_2_layer_call_and_return_conditional_losses_163861854\3�0
)�&
 �
inputs���������f
p
� "%�"
�
0���������f
� �
-__inference_dropout_2_layer_call_fn_163861841O3�0
)�&
 �
inputs���������f
p 
� "����������f�
-__inference_dropout_2_layer_call_fn_163861845O3�0
)�&
 �
inputs���������f
p
� "����������f�
O__inference_module_wrapper_2_layer_call_and_return_conditional_losses_163861802nEBDC?�<
%�"
 �
args_0���������f
�

trainingp "%�"
�
0���������f
� �
O__inference_module_wrapper_2_layer_call_and_return_conditional_losses_163861836nDEBC?�<
%�"
 �
args_0���������f
�

trainingp"%�"
�
0���������f
� �
4__inference_module_wrapper_2_layer_call_fn_163861748aEBDC?�<
%�"
 �
args_0���������f
�

trainingp "����������f�
4__inference_module_wrapper_2_layer_call_fn_163861782aDEBC?�<
%�"
 �
args_0���������f
�

trainingp"����������f�
K__inference_sequential_2_layer_call_and_return_conditional_losses_163861346v !EBDC56>�;
4�1
'�$
dense_8_input���������R
p 

 
� "&�#
�
0����������
� �
K__inference_sequential_2_layer_call_and_return_conditional_losses_163861407v !DEBC56>�;
4�1
'�$
dense_8_input���������R
p

 
� "&�#
�
0����������
� �
K__inference_sequential_2_layer_call_and_return_conditional_losses_163861570o !EBDC567�4
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
K__inference_sequential_2_layer_call_and_return_conditional_losses_163861631o !DEBC567�4
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
0__inference_sequential_2_layer_call_fn_163860794i !EBDC56>�;
4�1
'�$
dense_8_input���������R
p 

 
� "������������
0__inference_sequential_2_layer_call_fn_163861298i !DEBC56>�;
4�1
'�$
dense_8_input���������R
p

 
� "������������
0__inference_sequential_2_layer_call_fn_163861461b !EBDC567�4
-�*
 �
inputs���������R
p 

 
� "������������
0__inference_sequential_2_layer_call_fn_163861522b !DEBC567�4
-�*
 �
inputs���������R
p

 
� "������������
'__inference_signature_wrapper_163861662� !EBDC56G�D
� 
=�:
8
dense_8_input'�$
dense_8_input���������R"4�1
/
dense_11#� 
dense_11����������