import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams[u'font.sans-serif'] = ['simhei']
mpl.rcParams['axes.unicode_minus'] = False


arrow_args = dict(arrowstyle="<-", color='green')
leafNode = dict(boxstyle="round4", fc="0.9", color='red')
arrow_args = dict(arrowstyle="<-", color='green')


def countLeaf(root,deep):
    root.deep = deep
    res = 0
    if root.v=='好瓜' or root.v=='坏瓜':   # 说明此时已经是叶子节点了，所以直接返回
        res += 1
        return res,deep
    curdeep = deep             # 记录当前深度
    for i in root.children:    # 得到子树中的深度和叶子节点的个数
        a,b = countLeaf(i,deep+1)
        res += a
        if b>curdeep: curdeep = b
    return res,curdeep


def giveLeafID(root,ID):         # 给叶子节点编号
    if root.v=='好瓜' or root.v=='坏瓜':
        root.ID = ID
        ID += 1
        return ID
    for i in root.children:
        ID = giveLeafID(i,ID)
    return ID


def plotNode(nodeTxt,centerPt,parentPt,nodeType):     # 绘制节点
    plt.annotate(nodeTxt,xy = parentPt,xycoords='axes fraction',xytext=centerPt,
                 textcoords='axes fraction',va="center",ha="center",bbox=nodeType,
                 arrowprops=arrow_args)
 
def dfsPlot(root):
    if root.ID==-1:          # 说明根节点不是叶子节点
        childrenPx = []
        meanPx = 0
        for i in root.children:
            cur = dfsPlot(i)
            meanPx += cur
            childrenPx.append(cur)
        meanPx = meanPx/len(root.children)
        c = 0
        for i in root.children:
            nodetype = leafNode
            if i.ID<0: nodetype=decisionNode
            plotNode(i.v,(childrenPx[c],0.9-i.deep*0.8/deep),(meanPx,0.9-root.deep*0.8/deep),nodetype)
            plt.text((childrenPx[c]+meanPx)/2,(0.9-i.deep*0.8/deep+0.9-root.deep*0.8/deep)/2,i.title)
            c += 1
        return meanPx
    else:
        return 0.1+root.ID*0.8/(cnt-1)


def plot(myDecisionTreeRoot):
    decisionNode = dict(boxstyle = "sawtooth",fc = "0.9",color='blue')
    
    fig = plt.figure(1,facecolor='white')
    rootX = dfsPlot(myDecisionTreeRoot)
    plotNode(myDecisionTreeRoot.v,(rootX,0.9),(rootX,0.9),decisionNode)
    plt.show()