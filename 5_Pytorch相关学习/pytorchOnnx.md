# onnx

- 准备pytorch数据

```python
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

# Loading and normalizing the data.
# Define transformations for the training and test sets
transformations = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# CIFAR10 dataset consists of 50K training images. We define the batch size of 10 to load 5,000 batches of images.
batch_size = 10
number_of_labels = 10 

# Create an instance for training. 
# When we run this code for the first time, the CIFAR10 train dataset will be downloaded locally. 
train_set =CIFAR10(root="./data",train=True,transform=transformations,download=True)

# Create a loader for the training set which will read the data within batch size and put into memory.
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
print("The number of images in a training set is: ", len(train_loader)*batch_size)

# Create an instance for testing, note that train is set to False.
# When we run this code for the first time, the CIFAR10 test dataset will be downloaded locally. 
test_set = CIFAR10(root="./data", train=False, transform=transformations, download=True)

# Create a loader for the test set which will read the data within batch size and put into memory. 
# Note that each shuffle is set to false for the test loader.
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
print("The number of images in a test set is: ", len(test_loader)*batch_size)

print("The number of batches per epoch is: ", len(train_loader))
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
```

- 定义网络

```python
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

# Define a convolution neural network
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=5, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=5, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(12)
        self.pool = nn.MaxPool2d(2,2)
        self.conv4 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(24)
        self.conv5 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=5, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(24)
        self.fc1 = nn.Linear(24*10*10, 10)

    def forward(self, input):
        output = F.relu(self.bn1(self.conv1(input)))      
        output = F.relu(self.bn2(self.conv2(output)))     
        output = self.pool(output)                        
        output = F.relu(self.bn4(self.conv4(output)))     
        output = F.relu(self.bn5(self.conv5(output)))     
        output = output.view(-1, 24*10*10)
        output = self.fc1(output)

        return output

# Instantiate a neural network model 
model = Network()
```

- 定义损失函数

```python
from torch.optim import Adam
 
# Define the loss function with Classification Cross-Entropy loss and an optimizer with Adam optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
```

- 使用训练数据训练模型

```python
from torch.autograd import Variable

# Function to save the model
def saveModel():
    path = "./myFirstModel.pth"
    torch.save(model.state_dict(), path)

# Function to test the model with the test dataset and print the accuracy for the test images
def testAccuracy():
    
    model.eval()
    accuracy = 0.0
    total = 0.0
    
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            # run the model on the test set to predict labels
            outputs = model(images)
            # the label with the highest energy will be our prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()
    
    # compute the accuracy over all test images
    accuracy = (100 * accuracy / total)
    return(accuracy)


# Training function. We simply have to loop over our data iterator and feed the inputs to the network and optimize.
def train(num_epochs):
    
    best_accuracy = 0.0

    # Define your execution device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The model will be running on", device, "device")
    # Convert model parameters and buffers to CPU or Cuda
    model.to(device)

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        running_acc = 0.0

        for i, (images, labels) in enumerate(train_loader, 0):
            
            # get the inputs
            images = Variable(images.to(device))
            labels = Variable(labels.to(device))

            # zero the parameter gradients
            optimizer.zero_grad()
            # predict classes using images from the training set
            outputs = model(images)
            # compute the loss based on model output and real labels
            loss = loss_fn(outputs, labels)
            # backpropagate the loss
            loss.backward()
            # adjust parameters based on the calculated gradients
            optimizer.step()

            # Let's print statistics for every 1,000 images
            running_loss += loss.item()     # extract the loss value
            if i % 1000 == 999:    
                # print every 1000 (twice per epoch) 
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 1000))
                # zero the loss
                running_loss = 0.0

        # Compute and print the average accuracy fo this epoch when tested over all 10000 test images
        accuracy = testAccuracy()
        print('For epoch', epoch+1,'the test accuracy over the whole test set is %d %%' % (accuracy))
        
        # we want to save the model if the accuracy is the best
        if accuracy > best_accuracy:
            saveModel()
            best_accuracy = accuracy
```

- 使用测试数据测试模型

```python
import matplotlib.pyplot as plt
import numpy as np

# Function to show the images
def imageshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# Function to test the model with a batch of images and show the labels predictions
def testBatch():
    # get batch of images from the test DataLoader  
    images, labels = next(iter(test_loader))

    # show all images as one image grid
    imageshow(torchvision.utils.make_grid(images))
   
    # Show the real labels on the screen 
    print('Real labels: ', ' '.join('%5s' % classes[labels[j]] 
                               for j in range(batch_size)))
  
    # Let's see what if the model identifiers the  labels of those example
    outputs = model(images)
    
    # We got the probability for every 10 labels. The highest (max) probability should be correct label
    _, predicted = torch.max(outputs, 1)
    
    # Let's show the predicted labels on the screen to compare with the real ones
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] 
                              for j in range(batch_size)))
```



## 将pytorch模型转换为ONNX

通过pytorch训练的模型文件为`.pth`, 若要将其与Windows ML应用集成, 需要将模型转为ONNX格式

## 导出模型

要导出模型, 需要使用`torch.onnx.export()`函数, 此函数执行模型, 并记录用于计算输出的运算符的跟踪

1. 将main函数上方的以下代码复制到`PyTorchTraining.py`文件中

   ```python
   #main.py
   import torch.onnx
   
   #Function to Convert to ONNX
   def convert_ONNX():
       #set the model to inference mode
       model.eval()
       
       #let;s create a dummy input tensor
       dummy_input = torch.randn(1,input_size, requires_grad=True)
       #Export the model
       torch.onnx.export(model,	#model being run
                         dummy_input,	#model input (or a tuple for mutiple inputs)
                         "ImageClassifier.onnx",	#where to save the model
                         export_params=True,	#store the trained parameter weights inside the mode file
                         opset_version=10,	#the onnx version to export the model to
                         do_constant_folding=True,	#whether to execute constant folding for optimization
                         input_names=["modelInput"],	#the model's input names
                         output_names=["modelOutput"],	#the model's output names
                         dynamic_axes={"modelInput":{0:"batch_size"},	#variable length axes
                                      "modelOutput":{0:"batch_size"}}
                         )
       print(" ")
       print("Model has been convert to ONNX")
   ```

   在导出模型之前, 必须调用`model.eval()`或`model.train(False)`, 因为这会将模型设置为"推理"模式, 这是必须的, 因为`dropout`或`batchnorm`等运算符在推理和训练模式下的行为有所不同.

2. 要运行到ONNX的转换, 请将对转换函数的调用添加到main函数. 无需再次训练模型, 因此我们将注释一些需要运行的函数, main函数如下

   ```python
   if __name__ == "__main__":
       #let's build our model
       #train(5)
   	#print("Finished Training")
       
       #Test which classes performed well
       #testAccuracy()
       
       #Let's load the model we just created and test the accuracy per label
       model = Network()
       path = "myFirstModel.pth"
       model.load_state_dict(torch.load(path))
       
       #Test with batch of images
       #testBatch()
       #Test how the classes performed
       #testClasses()
       
       #Conversion to ONNX
       Convert_ONNX()
   ```

- 结果

  ![image-20211202163439583](E:\kuisu\typora\Python学习记录\Pytorch学习记录.assets\onnxShowStruct.png)

## pytorch2onnx

在pytorch中转换onnx的模型

```python
torch.onnx.export(model,args,f,export_params=True,verbose=False,input_names=None,output_names=None,do_constant_folding=True,dynamic_axes=None,opset_version=9)
```

| name                                         | descript                                                     |
| -------------------------------------------- | ------------------------------------------------------------ |
| model                                        | torch.nn.Module (要导出的模型)                               |
| args(tuple or tensor)                        | 模型的输入参数. 注意tuple的最后参数为dict. 输入参数只需要满足shape正确. |
| f:file object or string                      | 转换输出的模型的位置, 如"yolov3.onnx"                        |
| export_params: bool, default=True            | true表导出trained model, 否则为untrained model. 默认即可     |
| verbose: bool, default=False                 | true表打印调试信息                                           |
| input_names: list of string, default=None    | 指定输入节点名称                                             |
| output_names: list of string, default=None   | 指定输出节点名称                                             |
| do_constant_folding:bool, default=True       | 是否使用常量折叠, 默认即可                                   |
| dynamic_axes: dict<string, dict<int,string>> | 有时模型的输入输出是可变的, 如RNN, 或者输入输出图像的batch是可变的, 这时可以通过dynamic_axies来指定输入tensor的哪些参数可变. 如input的shape为(b,3,h,w)其中b, h, w可变.<br>1. 仅list(int): dynamic_axes={'input':[0,2,3],"output":[0,1]}<br />2. 仅dict<int, string><br />dynamic_axes={"input":{0:"batch",2:"height",3:"width"},"output":{0:"batch",1:"c"}}<br />3. mixed<br />dynamic_axes={"input":{0:"batch",2:"height",3:"width"},"output":[0,1]} |
| opset_version: int, default=9                | 指定onnx的opset版本, 版本过低的话, 不支持upsample等操作      |

- 实例

```python
import torch
import torch.nn as nn
import onnx
import numpy as np

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.conv1=nn.Conv2d(3,3,kernel_size=3,stride=2,padding=1)
        
    def forward(self,x,y):
        result1=self.conv1(x)
        result2=self.conv1(y)
        return result1, result2

model = Model()
model.eval()#若存在BN, dropout层, 则一定要用eval

input_names=["input_0","input_1"]
output_names=["output_0","output_1"]

x=torch.randn((1,3,12,12))
y=torch.randn((1,3,6,6))

torch.onnx.export(model,(x,y),'model.onnx',input_names=input_names,output_names=output_names,dynamic_axes={"input_0":[0],"output_0":[0]})
#指定input_0和output_0的batch可变
```

## trace and script

pytorch是动态计算图,onnx是静态图, pytorch转为静态图, 有两种方法`torch.jit.trac`和`torch.jit.script`

- `torch.jit.trace`: 给定模型一个输入(只要求输入的shape正确), 开始执行一次前向传播, 会记录过程中的所有操作. 缺点是race将不会捕获根据输入数据而改变的行为. 如if语句, 只会记录执行的那一条分支, 同样, for循环的次数, 导出与跟踪运行完成相同的静态图. 如果要使用动态控制流导出模型, 则需要使用torch.jit.script
- `torch.jit.script`: 真正的去编译, 去做语法分析. 因此可以使用if等控制动态流.

`torch.jit.script`中可能存在的问题

python是弱类型语言, 要进行编译, 需要知道变量类型, 要么程序员显示地指定变量类型, 要么python自动判断. 但自动判断不一定正确, `torch.jit.script`会把函数的所有输入都看成tensor类型, 并且令人诟病的是它会将list[tensor]类型看成tensor. 其解决方法如下

- 面对把函数所有输入都看成tensor类型的问题: 显示指定变量类型

```python
def head_process(features,model_image_size,num_anchors,anchors):
    #type: (List[torch.Tensor],Tuple(int,int),int,torch.Tensor)->torch.Tensor
```

- 面对把list[tensor]类型看成tensor的问题: 通过torch.jit.annotate指定变量类型

```python
prediction = torch.jit.annotate(List[troch.Tensor],[])
```

## 使用onnxruntime运行模型

- 检测模型

```python
import onnx
model = onnx.load('model.onnx')#load onnx
onnx.checker.check_model(model)#检查生成模型是否错误
```

使用onnxruntime推理, 注意输入的numpy的type一定要np.float32, 因为torch模型是float32, 这里要保持一致

```python
import onnxruntime as ort
ort_session = ort.InferenceSession("model.onnx")#创建一个推理session
x = np.random.randn(3,3,12,12).astype(np.float32)#输入类型一定要np.float32
y=np.random.randn((1,3,6,6)).astype(np.float32)

outputs = ort_session.run(None,{"input_0":x, "input_1":y})

'''
run()
args有两个参数: output_name, input
1. output_names: tuple of string , default=None
	若为None, 则按顺序输出所有的output, 即返回[output_0,output_1]
	若为['output_1',"output_0"],则返回[output_1,output_0]
	若为["output_0"],则仅返回[output_0:tensor]
2. input:dixt
	可以通过ort_session.get_input()[0].name,ort_session.get_input()[1].name获得名称
	其中key值要求与torch.onnx.export中设定的一致.
return:返回一个由output_names指定的list

'''
ort_inputs = {ort_session.get_inputs()[0].name:x,ort_session.get_inputs()[1].name:y}
ort_outs = ort_session.run(None, ort_inputs)
# 通过 get_inputs()[i].name来获取输入的名称
```

- 用`np.testing.assert_allclose`检查pytorch模型和导出的onnx模型的输出是否一致, 如果不一致, 会报错

```python
x = torch.randn(size=(batchsize,3,h,w),dtype=torch.float32).to(self.device)
model.eval()#pytorch model
torch.onnx.export(model,x,save_onnx_path,opset_version=11,input_names=["input"],output_names=['output'])

with torch.no_grad():
    torch_out = model(x)

import onnxruntime
ort_session = onnxruntime.InferenceSession(save_onnx_path)
#compute onnx runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name:x.cpu().numpy()}
ort_outs = ort_session.run(None,ort_inputs)

np.testing.assert_allclose(torch.out.cpu().numpy(),ort_outs[0],rtol=1e-03,atol=1e-05)
print("Exported model has been tested with ONNXRuntime, and the result looks good!")
```

如果测试模型运行时间发现变慢, 很可能是没有安装gpu版的onnxruntime

## tracing vs scripting

Internally, `torch.onnx.export()` requires a `torch.jit.ScriptModule` rather than a `torch.nn.Module`. if the paased-in model is not already a `ScriptModule`, `export()` weill use tracing to convert it to one:

- **Tracing**: if `torch.onnx.export()` is called with a Module that not already a `ScriptModule`, it first does the equivalent of `torch.jit.trace()`, which executes the model once with the given `args` and recordds all operations that happen during that execution. This means that if your model is dynamic, e.g., changes behavior depending on input data, the exported model will not capture this dynamic behavior. Similarly, a trace is likely to be valid only for a specific input size. we recommend examining the exported model and making sure the operators look reasonable. tracing will unroll loops and if statements, exporting a static graph that is exactly the same as the traced run. if you want to export your model with dynamic control flow, you will need to use scripting
- **Scripting**: compiling a model via scripting preserves dynamic control flow and is valid for inputs of different size. to use scripting
  - Use `torch.jit.script()` to produce a `ScriptModule`
  - Call `torch.onnx.export()` with `ScriptModule` as the model, and set the `example_outputs` arg. this required so that the type and shapes of the outputs can be captured without executing the model

### Avoding Pitfalls

#### Avoid numpy and build-in python types

Pytorch model can be written using numpy or python types and functions, but during `tracing`, any variables of numpy or python types (rather ran torch.Tensor) are converted to constats, which will produce the wrong result if those values should change depending on the inptus.

For example, rather than using numpy function on numpy.ndarrys:

```python
#Bad! will be replaed with constants during tracing:
x,y = np.random.randn(1,2), np.random.rand(1,2)
np.concatenate((x,y),axis=1)
```

use torch operators on torch.Tensor

```python
#Good! Tensor operations with be captured during tracing
x,y = torch.randn(1,2),torch.randn(1,2)
torch.cat((x,y),dim=1)
```

and rather than using `torch.Tensor.item()` (which converts a Tensor to a Python build-in number)

```python
#Bad! y.item() will be replaced with a constant during tracing
def forward(self,x,y):
    return x.reshape(y.item(),-1)
```

Use torch's support for implicit casting of single-element tensors:

```python
#Good! y will be preserved as variable during tracing
def forward(self,x,y):
    return x.reshape(y,-1)
```

#### Avoid Tensor.data

using the Tensor.data field can produce an incorrect trace and therefore an incorrect ONNX graph. Use `torch.Tensor.detach()` instead.

### Limitations

#### Types

- onoy torch.Tensors, numeric types that can be trivially convert to torch.Tensors (e.g. float, int), and tuples and lists of those types are supported as model inputs or outputs. Dict and str input and outputs are accepted in tracing model, but:
  - Any computation that depends on the value of dict or a str input will be replaced with the constant value seen during the one traced executioni
  - any output that is a dict will be silently replaced with a flatten sequence of its value (keys will be removed.) E.g. `{"foo": 1, "bar":2}` becomes `(1,2)`
- certain operations involving tupes and lists are not supported in `scripting` mode due to limited support in ONNX for nested sequences. In particular appending a tupel to a list is not supported. In tracing mode, the nested squences will be flattened automatically during the  tracing

#### Differences in operator implementations

Due to differences in iimplementations of operators, running the exported model on different runtimes may produce different results from each other or from pytorch. Normally these differences are numerically small, so this should only be a concern if your application is sentitive to these small differents

#### Unspported Tensor indexing patterns

Tensor indexing patterns that cannot be exported are listed below, if you are experiencing issues exporting a model that does not include any of the unsupported patterns below, please double check that you are exporting with the lasted `opset_version`

- Reads/Gets

When indexing into a tensor for reading, the following patterns are not supported:

```python
#Tensor indices that includes negative values
data[torch.tensor([[1,2],[2,-3]]), torch.tensor([-2,3])]
```

- Writes/sets

when indexing into a tensor for writing, the following pattern are not supported:

```python
#multiple tensor indices if any has rank >=2
data[torch.tensor([[1,2],[2,3]]),torch.tensor([2,3])]=new_data

#Multiple tensor indices that are not consecutive
data[torch.tensor([2,3]),:,torch.tensor([1,2])]=new_data

#tensor indices that includes negative values
data[torch.tensor([1,-2]),torch.tensor([-2,3])]=new_data
```

#### Adding support for operators

When exporting a model that includes unsupported operators, you'll see an error message like

```python
#RuntimeError: ONNX export failed: couldn't exporte operator foo
```

when that happens, you'll need to either change the model to not that operator, or add support for the operator. 

Adding support for operators requires contributing a change to pytorch's source code. See [contributing](https://github.com/pytorch/pytorch/blob/master/CONTRIBUTING.md) for general instructions on that, and below for specific instructions on the code changes required for supporting an operator.

During export, each node in the torchscript graph is visted in topological order. Upon visiting a node, the exporter tries to find a registered symbolic  functions for that node. Symbolic functionis are implemented in python. A symbolic function an op name `foo` would look something like

```python
def foo(
	g: torch._C.Graph,
    input_0:torch._C.Value,
    input_1:torch._C.Value
)->Union[None,torch._C.Value, List[Torch._C.Value]]:
    '''
    Modifies g (e.g, using 'g.op()'), adding the ONNX operations representing this pytorch function
    Args:
    	g (Graph): graph to write the ONNX representation into.
    	input_0 (Value): value representing the variables which contain the first input for this operator.
    	input_1 (Value): value representing the variables which contain the second input for this operator.
    
    Returns:
    	A value or list of values specifying the ONNX nodes that compute something equivalent to the original pytorch operator with the given inputs.
    	Returns None if it cannot be converted to ONNX
    '''
```

The `torch._C` type are Python wrappers around the types defined in C++ in [ir.h](https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/ir/ir.h)

The process for adding a symbolic function depends on the type of operator.



## torch.jit

### Mixing tracing and scripting

In many cases either tracing or scripting is an easier approach for converting a model to trochScript. Tracing and scripting can be composed to suit the particular requirements of a part of a model.

Scripted functions can call traced functions. This is particulary useful when you need to use control-flow around a simple feed-forward model. For instance the beam search of a sequence to sequence model will typically be written in script but can call an encoder module generated using tracing.

example (calling a traced function in script)

```python
import torch
def foo(x,y):
    return x*x+y

traced_foo = torch.jit.trace(foo, (torch.rand(3),torch.rand(3)))

@torch.jit.script
def bar(x):
    return traced_foo(x,x)
```

![image-20211203113811453](E:\kuisu\typora\Python学习记录\pytorchOnnx.assets\image-20211203113811453-16385026925471.png)

Traced functions can call script functions. This is useful when a small part of a model requires some control-flow even though most of the model is just a feed-forward network. Control-flow inside of a script function called by a traced function is preserved correctly

Example (call a script function in a traced function)

```python
import torch

@torch.jit.script
def foo(x,y):
    if x.max()>y.max():
        r=x
    else:
        r=y
    return r

def bar(x,y,z):
    return foo(x,y)+z

traced_bar = torch.jit.trace(bar,(torch.rand(3),torch.rand(3),torch.rand(3)))
```

![image-20211203114503551](E:\kuisu\typora\Python学习记录\pytorchOnnx.assets\image-20211203114503551-16385031046592.png)

This composition also works for `nn.Module` as well, where it can be used to generate a submodule using tracing that can be called fromthe methods of a script module.

- Example (using a traced module)

```python
import torch
import torchvision

class MyScriptModule(torch.nn.Module):
    def __init__(self):
        super(MyScriptModule,self).__init__()
        self.means = torch.nn.Parameter(torch.tensor([103.939,116.779,123.68]).resize_(1,3,1,1))
        self.resnet = torch.jit.trace(torchvision.models.resnet18(),torch.rand(1,3,224,224))
    def forward(self,input):
        return self.resnet(input-self.means)
my_script_module = torch.jit.script(MyScriptModule())
```

![image-20211203115617455](E:\kuisu\typora\Python学习记录\pytorchOnnx.assets\image-20211203115617455-16385037784823.png)

### TorchScript Language

TorchScript is a statically typed subset of Python, so many Python features apply directly to TorchScript. See the full TorchScrip Language Reference for details

| Type                 | Description                                                  |
| -------------------- | ------------------------------------------------------------ |
| Tensor               | A PyTorch tensor of any dtype, dimension, or backend         |
| Tuple[T0,T1,..,TN]   | A tuple containing subtypes (eg. Tuple[Tensor,Tensor])       |
| bool                 | A boolean value                                              |
| int                  | A scalar integer                                             |
| float                | A scalar floating point nuber                                |
| str                  | A string                                                     |
| List[T]              | A list of which all members are type T                       |
| Optional[T]          | A value which either None or Type T                          |
| Dict[K,V]            | A dict with key type K and value Type V. Only `str, int`, and float are allowed as key types |
| T                    | A torchScript Class                                          |
| E                    | A TorchScirpt Enum                                           |
| NamedTuple[T0,T1,..] | A collection.namedtuple tuple type                           |
|                      |                                                              |

#### Default Types

By default, all parameters to a TorchScript function are assumed to be Tensor.To Specify that an argument to a TorchScript function is another type, it is possible to use MyPy-Stype type annotation using the type listed above

```python
import torch

@torhc.jit.script
def foo(x,tup):
    #type: (int, Tuple[Tensor,Tensor])->Tensor
    t0,mt1=tup
    return t0+t1+x

print(foo(3,(torch.rand(3),torch.rand(3))))
```

it is also possible to annotate types with python3 type hints from the `typing` module

```python
import torch
from typing import Tuple

@torch.jit.script
def foo(x:int, tup:Tuple[torch.Tensor,torch.Tensor])->torch.Tensor:
    t0,t1=tup
    return t0+t1+x

print(foo(3,(torch.rand(3),torch.rand(3))))
```

- Example (type annotations for python 3)

```python
import torch
import torch.nn as nn
from typing import Dict, List, Tuple

class EmptyDataStructures(torch.nn.Module):
    def __init__(self):
        super(EmptyDataStructures, self).__init__()
    
    def forward(self, x:torch.Tensor)->Tuple[List[Tuple[int,float]],Dict[str,int]]:
        #this annotates the list to be a `List[Tuple[int,float]]`
        my_list: List[Tuple[int,float]]=[]
        for i in range(10):
            my_list.append((i,x.item()))

        my_dixt:Dict[str,int]={}
        return my_list, my_dict

x=torch.jit.script(EmpytDataStructures())
```

#### Optional Type Refinement

TorchScript will refine the type of a variable of type `Optional[T]` when a comparision to `None` is made inside the conditioinal of an in-statement or checked in `assert`. The compiler can reason about multiple `None` checks that are combined with `and, or, nor`. Refinement will also occur for else blocks of if-statements that are not explicitly written.

The `None` check must be within the if-statement's condition; assigning a `None` check to a variable and using it in the if-statement's condition will not refine the types of variable in the check. Only local variables will be refined, an attribute like `self.x` will not and must assigned to a local variable to be refined

Example (refining types on paramenters and locals)

```python
import torch
import torch.nn as nn
from typing import Optional

class M(nn.Module):
    z:Optional[int]
    def __init__(self,z):
        super(M,self).__init__()
            #If 'z' is None, Its type connot be infered, so it must be specified (above)
        self.z = z
    
    def forward(self,x,y,z):
        #type:(Optional[int],Optional[int],Optional[int])->int
        if x is None:
            x=1
            x=x+1
        #Refinement for an attribute by assigning it to a local
        z = self.z
        if y is not None and z is not None:
            x = y+z
        #refinement via an 'assert'
        assert z is not None
        x +=z
        return x
module = torch.jit.script(M(2))
module = torch.jit.script(M(None))
```

#### TorchScript Classes

Python classes can be used in TorchScript if they are annotated with `@troch.jit.script`,similar to how you would declare a torchscript function:

```python
@torch.jit.script
class Foo:
    def __init__(self,x,y):
        self.x =x
    def aug_add_x(self,inc):
        self.x += inc
```

this subset is restricted:

- All functions must be valid TorchScript functions (including `__init__`)

- Classes must be new-style classes, as we use `__new__()` to construct them with pybind11

- TorchScript classes are statically typed. Members can only be declared by assigining to self in the `__init__()` method

  for example, assigning to `self` outside of the `__init__()` method:

  ```python
  @torch.jit.script
  class Foo:
    def assign_x(self):
      self.x = torch.rand(2, 3)
  ```

  will reuslt in

  ```python
  RuntimeError:
  Tried to set nonexistent attribute: x. Did you forget to initialize it in __init__()?:
  def assign_x(self):
    self.x = torch.rand(2, 3)
    ~~~~~~~~~~~~~~~~~~~~~~~~ <--- HERE
  ```

- No expression except method definitions are allowed in the body of class

- No support for inheritance or any other polymorphism strategy. except for inheriting from `object` to specify a new-style class.

After a class is defined, it can be used in both torchscript and python interchangeably like any other torchsript type:

```python
#declare a torchscript class
@torch.jit.script
class Pair:
    def __init__(self,first, second):
        self.first = first
        self.second = second
        
@torch.jit.script
def sum_pair(p):
    #type:(Pair)->Tensor
    teturn p.first +p.second

p=Pair(torch.rand(2,3),torch.rand(2,3))
print(sum_pair(p))
```

#### torchScript Enums

Python enums can be used in TorchScript without any extra annotation or code"

```python
from enum import Enum
class Color(Enum):
    RED=1
    GREEN=2
@torch.jit.script
def enum_fn(x:Color,y:Color)->bool:
    if x==Color.RED:
        return True
    return x==y

```

After an enum is defined, it can be used in both TorchScript and python interchangeably like any other TorchScript type. The type of the values of an enum must be `int, float, str`. All values must be of the same type; heterogenous types for enum values are not supported



#### Name Tuples

Type produced by `collections.namedtuple` can be used in TorchScript

```python
import torch
import collections

Point = collections.namedtuple("Point",["x","y"])

@torch.jit.script
def total(point):
    #type: (Point)->Tensor
    return point.x+point.y

p = Point(x=torch.rand(3),y=torch.rand(3))
print(total(p))
```

#### Expressions

- Literals

  ```python
  True
  False
  None
  'string literals'
  "string literals"
  3 #interpreted as int
  3.4 #interpreted as a float
  ```

- List Construction

  An empty list is assumed have type `List[Tensor]`. The types of other list literals are derived from the type of the members

  ```python
  [3,4]
  []
  [torch.rand(3),torch.rand(4)]
  ```

- Tuple Constructioin

  ```python
  (3,4)
  (3,)
  ```

- Dict Construction

  An empty dict is assumed have type `Dict[str, Tensor]`. The types of other dict literals are derived from the type of the members.

  ```python
  {"hello":3}
  {}
  {'a':torch.randn(3),'b':torch.rand(4)}
  ```

- Arithmetic Operators

  ```python
  a + b
  a - b
  a * b
  a / b
  a ^ b
  a @ b
  ```

- Comparison Operators

  ```python
  a == b
  a != b
  a < b
  a > b
  a <= b
  a >= b
  ```

- Subscripts and slicing

  ```python
  t[0]
  t[-1]
  t[0:2]
  t[1:]
  t[:1]
  t[:]
  t[0, 1]
  t[0, 1:2]
  t[0, :1]
  t[-1, 1:, 0]
  t[1:, -1, 0]
  t[i:j, i]
  ```

- Function calls

  ```python
  import torch
  
  @torch.jit.script
  def foo(x):
      return x + 1
  
  @torch.jit.script
  def bar(x):
      return foo(x)
  ```

#### Method Calls

calls to methods of builtin types like tensor: `x.mm(y)`

On modules, methods must be compiled before they can be called. The TorchScript compiler recursively compiles methods it sees when compiling other methods. By default, compilation starts on the `forward` method. Any methods called by `forward` will be compiled, and any methods called by those methods, and so on. To start compilation at a method other than `forward`, use the `@torch.jit.export` decorator (`forward` implicitly is marked `@torch.jit.export`).

calling a submodule directly (e.g `self.resnet(input)`) is equivalent to calling its `forward` method (eg. `self.resnet.forward(input)`)

```python
import torch
import torch.nn as nn
import torchvision

class MyModule(nn.Module):
    def __init__(self):
        super(MyModule,self).__init__()
        means = torch.tensor([103.939, 116.779, 123.68])
        self.means = torch.nn.Parameter(means.resize_(1, 3, 1, 1))
        resnet = torchvision.models.resnet18()
        self.resnet=torch.jit.trace(resnet,torch.rand(1,3,224,224))
    
    def helper(self,input):
        return self.resnet(input-self.means)
    
    def forward(self,input):
        return self.helper(input)
    
    #since nothing in the model calls `top_level_method`, the compiler must be explicitly told to compile this method
    @torch.jit.export
    def top_level_method(self,input):
        return self.other_helper(input)
    
    def other_helper(self,input):
        return input+10

#'my_script_module' will have the compiled methods 'forward', 'helper'
#'top_level_method', and "other_helper"
my_script_module = torch.jit.script(MyModule())
```

#### Ternary Expressions

```python
x if x>y else y
```

#### For loops over tuple

These unrool the loop, generating a body for each member of tupoe. The budy must type-check correctly each member

```python
tup = (3, torch.rand(4))
for x in tup:
    print(x)
```

#### for loops over constant nn.ModuleList

To use a `nn.ModuleList` inside a compiled method, it must be marked constant by adding the name of the attribute to the `__constants__` list for the type. For loops over a `nn.ModuleList` will unroll the body of the loop at compile time, with each member of the constant module list.

```python
class SubModule(torch.nn.Module):
    def __init__(self):
        super(SubModule, self).__init__()
        self.weight = nn.Parameter(torch.randn(2))
    
    def forward(self, input):
        return self.weight + input
    
class MyModule(torch.nn.Module):
    __constants__ = ['mods']
    
    def __init__(self):
        super(MyModule,self).__init__()
        self.mods = torch.nn.ModuleList([SubModule() for i in range(10)])
    
    def forward(self,v):
        for module in self.mods:
            v = module(v)
        return v
m = torch.jit.script(MyModule())
```

#### Variable Resolution

TorchScript supports a subset of Python's variable resolution (i.e. scoping) rules. Local variables behave the same as in Python. Except for the restriction that a variable must bave the same type along all paths through a function. If a variable has a differren type on different branches of an if statement, it is an error to use it after the end of the if statement

Similarly, a variable is not allowed to be used if it is only defined along some paths through the function

Example

```python
@torch.jit.script
def foo(x):
    if x < 0:
        y=4
    print(y)
```

```python
Traceback (most recent call last):
  ...
RuntimeError: ...

y is not defined in the false branch...
@torch.jit.script...
def foo(x):
    if x < 0:
    ~~~~~~~~~
        y = 4
        ~~~~~ <--- HERE
    print(y)
and was used here:
    if x < 0:
        y = 4
    print(y)
          ~ <--- HERE...
```

Non-local variables are resolved to python values at compile time when the function is defined. These values are then converted into TorchScript values using the rules

### Use of Python Values

To make writing torchScript more convenient, we allow script code to refer to python values in the surrounding scope. For instance, any time there is a reference to `torch`, the TorchScript compiler is actually resolving it to the `torch` Python module when the function is declared. These Python values are not a first class part of torchScript. Instead they are de-sugared at compile-time into the primitive types that Torchscript supports. This depends on the dynamic type of the Python valued referenced when compilation occurs. This section describes the rules that are used when accessing Python values in TorchScript

#### Functions

TorchScript can call Python functions. This functionality is very useful when incrementally converting a model to TorchScript. The model can be moved function-by-function to TrochScript, leaving calls to Python functions in place. This way you can incrementally check the correctness of the model as you go 

> `torch.jit.is_scripting()`

Function that returns true when in compilation and false otherwise. This is useful especialy with the @unused decorator to leave code in you model that is not yet TorchScript compatible 

```python
import torch

@torch.jit.unused
def unsupported_linear_op(x):
    return x

def linear(x):
    if torch.jit.is_scripting():
        return torch.linear(x)
    else:
        return unsupported_linear_op(x)
```

#### Attribute lookup on python modules

TorchScript can lookup attributes on modules. Builtin functions like `torch.add` are accessed this way. This allows torchscript to call functions defined in other modules

- Python defined constants

  TorchScript also provides a way to use constants that are defined in python. These can be used to hard-code hyper-parameters into the function, or to define universal constant. There are two ways of specifiying that a python value should be treated as a constant.

  1. value looked up as attributes of a module are assumed to constant:

  ```python
  import match
  import torch
  
  @torch.jit.script
  def fn():
      return math.pi
  ```

  2. Attributes of a scriptModule can be marked constant by annotating them with `Final[T]`

  ```python
  import torch
  import torch.nn as nn
  
  class Foo(nn.Module):
      #"final" from the "typing_extensions" module can also be used
      a: torch.jit.Final[int]
      
      def __init__(self):
          super(Foo, self).__init__()
          self.a = 1+4
          
      def forward(self,input):
          return self.a + input
      
  f = torch.jit.script(Foo())
  ```

#### Module Attributes

The `torch.nn.Parameter` wrapper and `register_bugger` can be used to assign tensors to a module. Other values assigned to a module that is compiled will be added to the compiled module fi their types can be infered. All types available in TorchScript can be used as module attributes. Tensor attributes are semantically the same as buffers. The type of empty lists and dictioinaries and `None` values cannot be infered and must be specified via class annotations. If a type cannot be inferred and is not explicilty annotated, it will not be added as an attribute to the resulting scriptmoudle



```python
from typing import List, Dict
class Foo(nn.Module):
    #"words" is initialized as an empty list, so its type must be specified
    words: List[str]
        
    #The type could potentially be inferred if 'a_dict' (below) was not empty, but this annotation ensures 'some_dict' will be made into the proper type
    some_dict: Dict[str, int]
    
    def __init__(self, a_dict):
        super(Foo, self).__init__()
        self.words = []
        self.some_dict = a_dict
        
        #'int's can be inferred
        self.my_int = 10
        
    def forward(self, input):
        # type: (str)->int
        self.words.append(input)
        return self.some_dict[input]+self.my_int
f = torch.jit.script(Foo({'hi':}))
```

## C++_dnn

```c++
// DNN_YOLO_V4.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//#include "pch.h"

#include<opencv2/opencv.hpp>
#include<opencv2/dnn.hpp>
#include <iostream>
#include <fstream>
#include<istream>
#include<string>

using namespace std;
using namespace cv;
//using namespace dnn;

// 初始化参数
float confThreshold = 0.5; // 置信度阈值
float nmsThreshold = 0.4;  // 非极大值抑制(NMS)阈值
int inpWidth = 416;        // 网络输入图像宽度
int inpHeight = 416;       // 网络输入图像高度


						   // 加载类别名称文件
vector<string>classes;
// Load names of classes
string classesFile = "E:\\kuisu\\vsStudio\\OpenCV\\dnnYolo\\yolo\\coco.names";//字符串数组
ifstream ifs(classesFile.c_str());//*.c_str(), string转为连续的字符串


// 设置模型配置文件和权重
String config =  "E:\\kuisu\\vsStudio\\OpenCV\\dnnYolo\\model\\yolov3_tiny.cfg";
String weights = "E:\\kuisu\\vsStudio\\OpenCV\\dnnYolo\\model\\yolov3_tiny.weights";

// 加载网络
dnn::Net net = dnn::readNetFromDarknet(config, weights);



// 获取输出层名称
vector<String> getOutputsNames(const dnn::Net& net)
{
	static vector<String> names;
	if (names.empty())
	{
		//Get the indices of the output layers, i.e. the layers with unconnected outputs
		vector<int> outLayers = net.getUnconnectedOutLayers();

		//get the names of all the layers in the network
		vector<String> layersNames = net.getLayerNames();

		// Get the names of the output layers in names
		names.resize(outLayers.size());
		for (size_t i = 0; i < outLayers.size(); ++i)
			names[i] = layersNames[outLayers[i] - 1];
	}
	return names;
}

// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
{
	//Draw a rectangle displaying the bounding box
	rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 0, 255), 2);

	//Get the label for the class name and its confidence
	string label = format("%.2f", conf);
	if (!classes.empty())
	{
		CV_Assert(classId < (int)classes.size());
		label = classes[classId] + ":" + label;
	}

	//Display the label at the top of the bounding box
	int baseLine;
	Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.8, 1, &baseLine);
	top = max(top, labelSize.height);
	putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);
}

// Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess(Mat& frame, const vector<Mat>& outs)
{
	vector<int> classIds;
	vector<float> confidences;
	vector<Rect> boxes;

	for (size_t i = 0; i < outs.size(); ++i)
	{
		// Scan through all the bounding boxes output from the network and keep only the
		// ones with high confidence scores. Assign the box's class label as the class
		// with the highest score for the box.
		float* data = (float*)outs[i].data;
		for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
		{
			Mat scores = outs[i].row(j).colRange(5, outs[i].cols);//[c,x,y,w,h,c0,c1,..,c79]
			Point classIdPoint;//(x,y)
			double confidence;
			// Get the value and location of the maximum score
			minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);//(inputarray, out_minVal, out_maxVal, out_Point_minLoc, out_Point_maxLoc)
			if (confidence > confThreshold)
			{
				int centerX = (int)(data[0] * frame.cols);//cols->h
				int centerY = (int)(data[1] * frame.rows);//rows->w
				int width = (int)(data[2] * frame.cols);
				int height = (int)(data[3] * frame.rows);
				int left = centerX - width / 2;
				int top = centerY - height / 2;

				classIds.push_back(classIdPoint.x);
				confidences.push_back((float)confidence);
				boxes.push_back(Rect(left, top, width, height));//Rect 矩形类
			}
		}
	}

	// Perform non maximum suppression to eliminate redundant overlapping boxes with
	// lower confidences
	vector<int> indices;
	dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);//非极大值抑制
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		Rect box = boxes[idx];
		drawPred(classIds[idx], confidences[idx], box.x, box.y,
			box.x + box.width, box.y + box.height, frame);//class_id, confidence, x1,y1,x2,y2, image
	}
}

int main()
{
	Mat img = imread("E:\\kuisu\\vsStudio\\OpenCV\\dnnYolo\\person.jpg");
	if (img.empty())
	{
		cout << "Image read error, please check again!" << endl;
	}
	string line;
	while (getline(ifs, line))
	{
		classes.push_back(line);
	}
	net.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
	net.setPreferableTarget(dnn::DNN_TARGET_CPU);
	// Create a 4D blob from a frame.
	Mat blob;
	dnn::blobFromImage(img, blob, 1 / 255.0, Size(inpWidth, inpHeight), Scalar(0, 0, 0), true, false);

	//Sets the input to the network
	net.setInput(blob);

	// Runs the forward pass to get output of the output layers
	vector<Mat> outs;
	net.forward(outs, getOutputsNames(net));

	// Remove the bounding boxes with low confidence
	postprocess(img, outs);//nms->drawRect

	// Put efficiency information. The function getPerfProfile returns the
	// overall time for inference(t) and the timings for each of the layers(in layersTimes)
	vector<double> layersTimes;
	double freq = getTickFrequency() / 1000;
	double t = net.getPerfProfile(layersTimes) / freq;//use time
	string label = format("Inference time for a frame : %.2f ms", t);
	putText(img, label, Point(0, 20), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 0), 2);
	namedWindow("OpenCV_YoloV4_Demo", WINDOW_NORMAL);
	imshow("OpenCV_YoloV4_Demo", img);
	waitKey(0);

	return 0;
}
```

- [参考来源](https://blog.csdn.net/stq054188/article/details/108697929)

视频检测

```c++
int main()
{
  string line;
  while (getline(ifs, line))
  {
    classes.push_back(line);
  }
  VideoCapture cap("./cars.mp4");
  Mat frame;
  while (1)
  {
    if (!cap.isOpened())
    {
      cout << "Video open failed, please check!" << endl;
      break;
    }
    cap.read(frame);
    if (frame.empty())
    {
      cout << "frame is empty, please check!" << endl;
      break;
    }
    
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);
    // Create a 4D blob from a frame.
    Mat blob;
    blobFromImage(frame, blob, 1 / 255.0, Size(inpWidth, inpHeight), Scalar(0, 0, 0), true, false);
 
    //Sets the input to the network
    net.setInput(blob);
 
    // Runs the forward pass to get output of the output layers
    vector<Mat> outs;
    net.forward(outs, getOutputsNames(net));
 
    // Remove the bounding boxes with low confidence
    postprocess(frame, outs);
 
    // Put efficiency information. The function getPerfProfile returns the
    // overall time for inference(t) and the timings for each of the layers(in layersTimes)
    vector<double> layersTimes;
    double freq = getTickFrequency() / 1000;
    double t = net.getPerfProfile(layersTimes) / freq;
    string label = format("Inference time for a frame : %.2f ms", t);
    putText(frame, label, Point(0, 20), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 0), 2);
    namedWindow("OpenCV_YoloV4_Demo", WINDOW_NORMAL);
    imshow("OpenCV_YoloV4_Demo", frame);
    int c = waitKey(1);
    if (c == 27)
      break;
  }
  return 0;
}
```

- python版

```python
#参考: [OpenCV DNN模块官方教程](https://blog.csdn.net/stq054188/article/details/108697929)
import cv2

config =  "E:\\kuisu\\vsStudio\\OpenCV\\dnnYolo\\model\\yolov3_tiny.cfg"
weights = "E:\\kuisu\\vsStudio\\OpenCV\\dnnYolo\\model\\yolov3_tiny.weights"
classesFile = "E:\\kuisu\\vsStudio\\OpenCV\\dnnYolo\\yolo\\coco.names"
confThreshold=0.4
nmsThreshold=0.4

with open(classesFile, 'r') as f:
    classes = f.readlines()

net = cv2.dnn.readNetFromDarknet(config,weights)

def getOutputsNames(net):
    '''
    获取net的输出层
    :param net:
    :return:
    '''
    #Get the indices of the output layers, i.e. the layers with unconnected outputs
    outLayers = net.getUnconnectedOutLayers()

    #get the names of all the layers in the network
    layersNames = net.getLayerNames()

    #Get the names of the output layers in names
    names = []
    for idx in outLayers:
        name = layersNames[idx[0]-1]
        names.append(name)
    return names

def drawPred(classId, conf, left, top, right,bottom, image):
    '''绘制检测到的目标框'''
    #draw a rectangle displaying the bounding box
    cv2.rectangle(image,(left,top),(right,bottom),color=(0,0,255),thickness=2)

    # get the label for the class name and its confidence
    label = "{:.2f}".format(conf)
    # if len(classId) >0:
    assert classId<79
    label = classes[classId]+":"+label

    # Display the label at the top of the bounding box
    labelSize = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.8,1)
    top = max(top, labelSize[1])
    cv2.putText(image,label,(left,top),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)


def postProcess(img, outs):
    '''
    进行nms处理
    :param img: input image
    :param outs: [ndarray]
    :return:
    '''

    #scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence socres. Assign the box's class label as the class
    # with the highest score for the box
    classIds = []
    confidences = []
    boxes = []
    for i in range(len(outs)):
        for j in range(len(outs[i])):
            scores = outs[i][j][5:]
            minScore, maxScore, minPoint, maxPoint = cv2.minMaxLoc(scores)
            if maxScore >confThreshold:
                cx = int(outs[i][j][0]*img.shape[1])#img.shape->[h,w,c]
                cy = int(outs[i][j][1]*img.shape[0])
                w = int(outs[i][j][2]*img.shape[1])
                h = int(outs[i][j][3]*img.shape[0])
                left = int(cx - w/2)
                top = int(cy - h/2)
                classIds.append(maxPoint[1])
                confidences.append(maxScore)
                boxes.append([left,top,w,h])
    print(classIds)
    print(confidences)
    print(boxes)
    indices = cv2.dnn.NMSBoxes(boxes,confidences,confThreshold,nmsThreshold)
    for i in range(len(indices)):
        box = boxes[i]
        drawPred(classIds[i],confidences[i],box[0],box[1],box[2]+box[0],box[1]+box[3],img)
    return img


def main():
    img = cv2.imread("E:\\kuisu\\vsStudio\\OpenCV\\dnnYolo\\person1.jpg")

    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)#推理
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)#target

    # create a 4D blob from a frame
    blob = cv2.dnn.blobFromImage(img,1/255.0,(416,416),(0,0,0), True, crop=False)

    #sets the input to the network
    net.setInput(blob)

    #runs the forward pass to get output of the output layers
    outs = net.forward(getOutputsNames(net))
    img = postProcess(img,outs)

    # put efficiency information. The function getPerfProfile return the
    # Overall time for inference(t) and the timings for each of the layers (in layersTimes)
    t,_ = net.getPerfProfile()
    t=t * 1000.0 / cv2.getTickFrequency()
    label = "inference time for a frame: {:.2f}ms".format(t)
    cv2.putText(img, label,(0,20),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,0),2)
    cv2.namedWindow("OpenCV_yolov3_demo",cv2.WINDOW_NORMAL)
    cv2.imshow("OpenCV_yolov3_demo",img)
    cv2.waitKey()


if __name__ == '__main__':
    main()

```

