# 决策树学习笔记

## 绘制决策树，显示中文

在主目录的utils中，找到SimHei.ttf。

```python
import matplotlib

print(matplotlib.matplotlib_fname())

# /Users/xxx/anaconda3/lib/python3.8/site-packages/matplotlib/mpl-data/matplotlibrc

# 返回当前python环境，对应的matplotlib对应的目录
```

获取目录后，进行如下操作

```bash
# 往matplotlib中添加新字体
$ cp SimHei.ttf /Users/xxx/anaconda3/lib/python3.8/site-packages/matplotlib/mpl-data/fonts/ttf/

# 清空matplotlib字体缓存
$ cd ~/.matplotlib
$ rm -rf *.cache
$ rm -rf fontList.*.cache  # 这里我观察到，ubuntu与mac对应的缓存命名不同，视情况删除即可。
```
