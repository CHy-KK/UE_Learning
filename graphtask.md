[【UE引擎篇】Runnable、TaskGraph、AsyncTask、Async多线程开发指南 - 水曜日鸡的文章 - 知乎](https://zhuanlan.zhihu.com/p/463272214)
Graph Task首先是一个模板类，我们需要自定义一个用于执行task的类传入，该类主要需要用户重载的部分为DoTask()，用于定义该task需要完成的任务。

假设自定义的类为FWorkTask

现在我们需要有一个actor来承载这些task，或者说创建并执行FWorkTask类的实例。

假定这个actor叫ATestTaskGraphActor，我们创建一个CreateTask方法来创建FWorkTask。对于每一个task，我们需要考虑该task是否具有前置以及子task。

注：下文提到的<b>创建时</b>都指在actor的CreateTask方法中执行；在<b>执行时</b>指在FWorkTask的DoTask()中执行

* 前缀：一个task直到所有的前缀任务Prerequisites完成才会开始执行
* child task：一个task可以被分为很多个子task，主task和子task之间不存在同步关系，即任意顺序执行。但是只有子task被完成才能视为主task被完成。切记不要把子task理解为主task前缀，他们本身就是一个大task，只是被划分为了多个task，供更多的线程来执行。

那么前缀和child task是如何实现的？

每个task在创建时，出来提供其模板类，还要提供其前缀事件列表PrerequisiteEvents，graphtask内部会直到前缀任务完成的事件发生之后才开始执行自身的task。前缀任务在蓝图上的表示即，一个节点（函数）的输入依赖另一个节点的返回值。该返回值以及输入的接口类型可以自定义，但是其中需要包含一个FGraphEventRef类型的变量，然后在创建后缀节点时把该FGraphEventRef加入PrerequisiteEvents，然后创建后缀节点对应的task时传入PrerequisiteEvents。

对于childtask，ue4其实没有对childtask做任何实现，只是提供一个接口FGraphEventRef::DontCompleteUntil(FGraphEventRef EventToWaitFor)。

首先解释DontCompleteUntil函数，其实看函数名就能明白，在给定的参数任务完成之前，其调用者无法被置为完成状态。

然后是FGraphEventRef EventToWaitFor，看了前缀后我们知道EventToWaitFor其实是反应一个task是否完成的事件，当task完成该事件就会被激活。所以任何拥有child task的任务（childtask存放在自定义的FWorkTask中），可以在DoTask()中使用DontCompleteUntil函数传入子任务的完成事件。


关于TGraphTask类的实现，首先我们要直到他有两个辅助类，一个是FGraphEvent，其中包含了上面提到的FGraphEventRef，用来处理task之间事件通信等。另一个是FConstructor，用来创建TGraphTask。

我们着重说一下FConstructor。如果我们调用TGraphTask::CreateTask()得到的其实是一个FConstructor的对象，其内部封装了一个TGraphTask*和一个FGraphEventArray，分别表示拥有该对象的真正的graphtask，以及该task所有的前缀事件。同时如果我们想要拿到这个graphtask真正的实例以及其前缀事件，FConstructor也提供了两个接口：
```c++
template<typename...T>
TGraphTask* ConstructAndHold(T&&... Args);
template<typename...T>
FGraphEventRef ConstructAndDispatchWhenReady(T&&... Args);
```
可以看到这两个接口都是模板，这个T其实就是我们自定义的FWorkTask的构建参数，也就是说我们实际是在graphtask中构建了一个FWorkTask任务。关于他们是如何实现的请看源码，这里不多赘述。

总之，通过这种方式我们可以拿到graphtask本体以及其前缀事件，那么就可以将其写入我们在说创建task时提到的接口类中，自此我们完成了一个graphtask创建的循环。

那么我们怎么运行整个graphtask呢，其实也非常的简单，直接在蓝图的最开始放置运行节点，调用TGraphTask::Unlock()即可，然后在拥有childtask的DoTask()中循环解锁child task。

另外，如果不想加入多线程直接运行FWorkTask::DoWork()也是没有问题的

UE中对于graphtask的应用举例：一个渲染方面非常重要的宏：ENQUEUE_RENDER_COMMAND，作用是向渲染器线程入队渲染指令，文件路径：Engine\Source\Runtime\RenderCore\Public，就用到了graphtask来实现。这个宏会使用EnqueueUniqueRenderCommand函数去构建一个task，这个task的类型是TEnqueueUniqueRenderCommandType<TSTR, LAMBDA>，第一个模板参数是一个字符串，第二个模板参数是一个lambda函数。在EnqueueUniqueRenderCommand函数中，如果我们需要将渲染指令加入单独的渲染线程任务队列中，就需要构建这个task。然后在这个task的dowork中，会向传入的lambda函数传入一个RHICmdList参数，也就是说调用ENQUEUE_RENDER_COMMAND宏时，第二个参数传入的必须是一个以RHICmdList为参数的lambda函数。然后，我们在这个lambda宏中编写我们需要这一条渲染指令想要执行的渲染内容。至此，我们就成功将这个任务成功加入了渲染线程的graphtask中。不得不说ue中加入一条渲染指令真的要走过很多步骤，封装了很多层hhh

ENQUEUE_RENDER_COMMAND的作用：
* 游戏线程和渲染线程的数据通信
* ......