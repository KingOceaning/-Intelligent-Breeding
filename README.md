simulateBCx.py两个文件负责产生回交的基因型数据，这里就不多赘述了

sample_select.py包含一些功能函数，比如：sample_fliter筛选已导入目标基因的样本并计算相应的指标；两个计算汇报的函数reward_compute（PS后来被我舍弃了没怎么用过）；compute _pl_pr计算目标基因左右断开连锁的概率（这段代码貌似有一点点小问题，有时候算出来概率会是负数[>~<]）

RF_for_BC.py是用来做回交决策的模型（使用前记得改里面的script_prefix，实在懒得在每个函数的参数里加所以我就把它改成内部变量了，最好Ctrl+F全部替换）

关于几个ipynb文件：
里面有一些具体的调用实例代码量有点多（主要没删。。。）
如果您不幸动了sequence文件夹下的数据，就找我再发一份吧，或者您想感受一下generate_data_1.ipynb里面生成数据用的代码的威力也不是不可以（兴许可以找到一些优化代码运行速度的方式？PS：已经从以前100条/100min优化成了100条/70min）。
RF(20-60).ipynb里面是运行模型进行决策的代码，稍微改改就能直接跑，记得选一个好点的position。
reprocess.ipynb不用管，只是不想删。。。

关于几个json文件（不是sequence文件夹下的）：
都是基因位点，直接json.load就可以拿来随机抽取一个位点。

感谢帮忙！Y{^_^}Y
