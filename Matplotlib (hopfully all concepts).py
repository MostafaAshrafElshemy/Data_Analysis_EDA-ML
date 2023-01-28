#!/usr/bin/env python
# coding: utf-8

# In[3]:


#matplotlib 
#seaborn 
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np


# In[4]:


x = [1,2,4,5,6]
y = [3,4,8,15,12]

plt.plot(x,y)


# In[5]:


x = [1,2,4,5,6]
y = [2,4,8,10,12]

plt.title("this is a linear graph")

plt.plot(x,y)
plt.show()


# In[4]:


x = [1,2,4,5,6]
y = [2,4,8,10,12]

plt.title("this is a good graph", fontdict={'fontname': 'Comic Sans MS', 'fontsize': 18})
plt.xlabel("time")
plt.ylabel("price")
plt.plot(x,y)
plt.show()


# In[7]:


x = [1,2,4,5,6]
y = [2,4,8,10,12]

plt.plot(x,y, label = "y = 2x" , color = "blue")
plt.title("this is a good graph")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.legend()
plt.show()


# In[8]:


x = [1,2,4,5,6]
y = [2,4,8,10,12]

plt.plot(x,y, label = "y = 2x" , color = "black")
plt.title("this is a good graph")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.legend()
plt.show()


# In[12]:


x = [1,2,4,5,6]
y = [2,4,8,10,12]

plt.plot(x,y, label = "2x" , color = "red", linewidth = 2)
plt.title("this is a good graph")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.legend()
plt.show()


# In[13]:


x = [1,2,4,5,6,7]
y = [2,4,8,10,12,14]

plt.plot(x,y, label = "2x" , color = "black", linewidth = 1 , marker = "x")
plt.title("this is a good graph")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.legend()
plt.show()


# In[15]:


x = [1,2,4,5,6]
y = [2,4,8,10,12]

plt.plot(x,y, label = "2x" , color = "gray", linewidth = 2 , marker = "*", linestyle="--")
plt.title("this is a good graph")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.legend()
plt.show()


# In[16]:


x = [1,2,4,5,6]
y = [2,4,8,10,12]

plt.plot(x,y, label = "2x" , color = "black", linewidth = 2 , marker = "*", linestyle=":")
plt.title("this is a good graph")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.legend()
plt.show()


# In[17]:


x = [1,2,4,5,6]
y = [2,4,8,10,12]

plt.plot(x,y,'b>--' ,label = "2x")
plt.title("this is a good graph")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.legend()
plt.show()


# In[41]:


x = [1,2,4,5,6]
y = [2,4,8,10,12]

plt.plot(x,y,'go:' ,label = "2x")
plt.title("this is a good graph")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.legend()
plt.show()


# In[18]:


x = [0,1,2,4,5,6]
y = [0,2,4,8,10,12]
x2 = np.arange(0,4.5, 0.5) # 0 , 0.5,  1 , 1.5. 2,  2.5 , 3, 3.5, 4 
#x2 = [.5,1,1.5,2,2.5,3....]

plt.plot(x,y,'r*--' ,label = "equation 1")
plt.plot(x2, x2**2,'g>--' ,label = "equation 2")

plt.title("this is a good graph")
plt.xlabel("X axis")
plt.ylabel("Y axis")


plt.legend()
plt.show()


# In[22]:


x = [1,2,4,5,6]
y = [2,4,8,10,12]
x2 = [0, 0.5 , 1 , 1.5 , 2, 2.5 ,3 , 3.5] # 0 , 0.5,  1 , 1.5. 2,  2.5 , 3, 3.5

plt.plot(x,y,'r*--' ,label = "equation 1")
plt.plot(x2, [(x*x) for x in x2],'g>--' ,label = "equation 2")

plt.title("this is a good graph")
plt.xlabel("X axis")
plt.ylabel("Y axis")


plt.legend(loc= "best")
plt.show()


# In[24]:


get_ipython().run_line_magic('matplotlib', 'inline')
x = [1,2,4,5,6]
y = [2,4,8,10,12]
x2 = np.arange(0,4, 0.5)

plt.plot(x,y,'r*--' ,label = "2x")
plt.plot(x2, x2**2, label = "blue")

plt.title("this is a good graph")
plt.xlabel("X axis")
plt.ylabel("Y axis")

plt.xticks(np.arange(0,7))
plt.yticks(np.arange(0,13))
plt.legend()
plt.show()


# In[25]:


get_ipython().run_line_magic('matplotlib', 'inline')
x = [1,2,4,5,6]
y = [2,4,8,10,12]
x2 = np.arange(0,4, 0.5)

plt.plot(x,y,'r*--' ,label = "2x")
plt.plot(x2, x2**2, label = "blue")

plt.title("this is a good graph")
plt.xlabel("X axis")
plt.ylabel("Y axis")

plt.xticks([0,1,2, 2.5,3,4,5,6])
plt.yticks([i for i in range (0,7)] +[9])
plt.legend()
plt.show()


# In[28]:


get_ipython().run_line_magic('matplotlib', 'inline')
x = [1,2,4,5,6]
y = [2,4,8,10,12]
x2 = np.arange(0,4, 0.5)

plt.plot(x,y,'r*--' ,label = "2x")
plt.plot(x2, x2**2, label = "blue")

plt.title("this is a good graph")
plt.xlabel("X axis")
plt.ylabel("Y axis")

plt.xticks([0,1,2, 2.5,3,4,5,6])
plt.yticks(list(range(0,15,2)) +[9])
plt.legend()
plt.savefig("random.png")
plt.show()


# In[27]:


get_ipython().run_line_magic('matplotlib', 'inline')
x = [1,2,4,5,6]
y = [2,4,8,10,12]
x2 = np.arange(0,4, 0.5)

plt.plot(x,y,'r*--' ,label = "2x")
plt.plot(x2, x2**2, label = "blue")

plt.title("this is a good graph")
plt.xlabel("X axis")
plt.ylabel("Y axis")

plt.xticks([0,1,2, 2.5,3,4,5,6])
plt.yticks(list(range(0,15,2)) +[9])
plt.legend()
plt.savefig("random.jpg")
plt.show()


# In[29]:


get_ipython().run_line_magic('matplotlib', 'inline')
x = [1,2,4,5,6]
y = [2,4,8,10,12]
x2 = np.arange(0,4, 0.5)

plt.plot(x,y,'r*--' ,label = "2x")
plt.plot(x2, x2**2, label = "blue")

plt.title("this is a good graph")
plt.xlabel("X axis")
plt.ylabel("Y axis")

plt.xticks([0,1,2, 2.5,3,4,5,6])
plt.yticks(list(range(0,15,2)) +[9])
plt.legend()
plt.savefig("random.pdf")
plt.show()


# In[30]:


get_ipython().run_line_magic('matplotlib', 'inline')
x = [1,2,4,5,6]
y = [2,4,8,10,12]
x2 = np.arange(0,4, 0.5)

plt.plot(x,y,'r*--' ,label = "2x")
plt.plot(x2, x2**2, label = "blue")

plt.title("this is a good graph")
plt.xlabel("X axis")
plt.ylabel("Y axis")

plt.xticks([0,1,2, 2.5,3,4,5,6])
plt.yticks(list(range(0,15,2)) +[9])
plt.legend()
plt.savefig("random2.jpg" ,dpi = 250)
plt.show()


# In[32]:


# if we want to change the size of all figures to be a new fixed size we use rc parameters
plt.rcParams["figure.figsize"] = (10, 10) 


# In[33]:


#surronding equations by $equation$ and adding r at first visualize the equation in mathematical way 
x = [1,2,3,4,5]
y = [1,4,9,16,25]
plt.title(r"$y= x^2$")
#plt.title(r"$y= x_i$")
#plt.title(r"$y= \cos(\pi . x)$")
plt.plot(x,y)
plt.show()
# plt.title(r"$y= \cos(\pi x)$")
# more : https://matplotlib.org/stable/tutorials/text/mathtext.html


# In[36]:


# legend
x = [1,2,3,4,5]
y = [1,4,9,16,25]
plt.title(r"$y= x^2$")
plt.plot(x,y, label =r"$y= x^2$" )
plt.legend(bbox_to_anchor= (1.05, 1)) #you can also add loc

# plt.legend(ncol = 2)
# plt.legend(labeelcolor=["blue", "orange"])
# we can add also bold size 
#reomove legned border = frameon = False
# add tiltle title= "Ahmed"


# In[39]:


# legend
x = [1,2,3,4,5]
y = [1,4,9,16,25]
z = [1,4,6,8,12]
plt.title(r"$y= x^2$")
plt.plot(x,y, label =r"$y= x^2$" )
plt.plot(x,z, label =r"$y= x^2$" )
plt.legend(ncol = 2)
plt.plot()
plt.show()
# plt.legend(labeelcolor=["blue", "orange"])
# we can add also bold size 
#reomove legned border = frameon = False
# add tiltle title= "Ahmed"


# In[40]:


# legend
x = [1,2,3,4,5]
y = [1,4,9,16,25]
z = [1,4,6,8,12]
plt.title(r"$y= x^2$")
plt.plot(x,y, label =r"$y= x^2$", color = "green")
plt.plot(x,z, label =r"$y= x^2$" ,color = "grey")
plt.legend(labelcolor=["black", "red"])
plt.plot()
plt.show()
# we can add also bold size 
#reomove legned border = frameon = False
# add tiltle title= "Ahmed"


# In[101]:


# legend
x = [1,2,3,4,5]
y = [1,4,9,16,25]
z = [1,4,6,8,12]
plt.title(r"$y= x^2$")
plt.plot(x,y, label =r"$y= x^2$" )
plt.plot(x,z, label =r"$y= x^2$" )
plt.legend(labelcolor=["black", "red"], frameon = False)
plt.plot()
plt.show()
# we can add also bold size 
#reomove legned border = frameon = False
# add tiltle title= "Ahmed"


# In[41]:


# legend
x = [1,2,3,4,5]
y = [1,4,9,16,25]
z = [1,4,6,8,12]
plt.title(r"$y= x^2$")
plt.plot(x,y, label =r"$y= x^2$" )
plt.plot(x,z, label =r"$y= x^2$" )
plt.legend(labelcolor=["black", "red"], title = "key map")
plt.plot()
plt.show()
# we can add also bold size 
#reomove legned border = frameon = False
# add tiltle title= "Ahmed"


# In[42]:


# # remove the border around the figures
x = [1,2,3,4,5]
y = [1,4,9,16,25]
z = [1,4,6,8,12]
plt.title(r"$y= x^2$")
plt.plot(x,y, label =r"$y= x^2$" )
plt.plot(x,z, label =r"$y= x^2$" )
plt.legend(labelcolor=["black", "red"], title = "key map")
ax = plt.gca() # get current axis 
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#ax.set_frame_on(False)
plt.show()
# # if we want to do that we change rc params
# plt.rcParams["axes.spines.top"] = False
# plt.rcParams["axes.spines.right"] = False
# task : do it with seaborn


# In[43]:


# # remove the border around the figures
x = [1,2,3,4,5]
y = [1,4,9,16,25]
z = [1,4,6,8,12]
plt.title(r"$y= x^2$")
plt.plot(x,y, label =r"$y= x^2$" )
plt.plot(x,z, label =r"$y= x^2$" )
plt.legend(labelcolor=["black", "red"], title = "key map")
ax = plt.gca() # get current axis 
#ax.spines['top'].set_visible(False)
ax.set_frame_on(False)
plt.show()


# In[44]:


# # remove the border around the figures
x = [1,2,3,4,5]
y = [1,4,9,16,25]
z = [1,4,6,8,12]
plt.title(r"$y= x^2$")
plt.plot(x,y, label =r"$y= x^2$" )
plt.plot(x,z, label =r"$y= x^2$" )
plt.legend(labelcolor=["black", "red"], title = "key map")
ax = plt.gca() # get current axis 
#ax.spines['top'].set_visible(False)
ax.set_frame_on(False)
plt.xticks([])
plt.yticks([])
plt.show()


# 
# ### 3D plots

# In[6]:


plt.rcParams['figure.figsize'] = (8,6)
np.random.seed(31)
mu = 3
n=50

x = np.random.normal(mu, 1, size=n)
y = np.random.normal(mu, 1, size=n)
z = np.random.normal(mu, 1, size=n)
ax = plt.axes(projection='3d')
ax.scatter3D(x, y, z)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.view_init(45, 215)


# In[7]:


plt.rcParams['figure.figsize'] = (8,6)


# In[9]:


ax = plt.axes(projection='3d');


# In[115]:


# ax = plt.axes(projection='3d')
ax.scatter3D(x, y, z);


# In[10]:


ax = plt.axes(projection='3d')
ax.scatter3D(x, y, z, s=100)

# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z');


# In[11]:


ax = plt.axes(projection='3d')
ax.scatter3D(x, y, z, s=100)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

ax.view_init(45, 215);


# In[12]:


omega = 2

z_line = np.linspace(0, 10, 100)
x_line = np.cos(omega*z_line)
y_line = np.sin(omega*z_line)


# In[13]:


ax = plt.axes(projection='3d')
ax.plot3D(x_line, y_line, z_line, lw=4);


# In[20]:


# 3D wireframe 
# N = 10
# x_values = np.linspace(-5, 5, N)
# y_values = np.linspace(-5, 5, N)


# In[18]:


# X,Y = np.meshgrid(x_values, y_values)
# Z = function_z(X, Y)


# In[21]:


ax = plt.axes(projection='3d')
ax.plot_wireframe(X, Y, Z, color='black', lw=3);


# In[ ]:


x = [1,2,4,5,6]
y = [2,4,8,10,12]
x2 = np.arange(0,4, 0.5)


plt.figure(figsize=(8,4), dpi=250)

plt.plot(x,y,'r*--' ,label = "2x")
plt.plot(x2, x2**2)

plt.title("this is a good graph")
plt.xlabel("X axis")
plt.ylabel("Y axis")

plt.xticks(np.arange(1,15 ))

plt.legend()
plt.savefig("goodgraph.jpg")
plt.show()


# In[46]:


x = [1,2,4,5,6]
y = [2,4,8,10,12]
x2 = np.arange(0,4, 0.5)


plt.figure(figsize=(10,4), dpi=250)

plt.plot(x,y,'r*--' ,label = "2x")
plt.plot(x2, x2**2)

plt.title("this is a good graph")
plt.xlabel("X axis")
plt.ylabel("Y axis")

plt.xticks([0,1,2,3,4,5,6,7,8,9,10])

plt.legend()
plt.show()


# ### bar chart 

# In[47]:


lables = ["A", "B", "C"]
values = [1,4 ,2]
plt.bar(lables, values, color = "#8f285b")
plt.yticks(np.arange(0,5,1))
plt.show()


# In[48]:


lables = ["A", "B", "C"]
values = [1,4 ,2]
bars = plt.bar(lables, values, color = "#034dad")
bars[0].set_hatch('/')
bars[1].set_hatch('o')
bars[2].set_hatch('*')
plt.show()


# In[49]:


lables = ["A", "B", "C"]
values = [1,4 ,2]
plt.bar(lables, values, color = "#034dad")[0].set_hatch("/")
plt.show()


# In[50]:


lables = ["A", "B", "C"]
values = [1,4 ,2]
bars = plt.bar(lables, values)
bars[0].set_hatch('o')
plt.show()


# In[135]:


lables = ["A", "B", "C"]
values = [1,4 ,2]
bars = plt.bar(lables, values , color = "#8f285b")
patterns = ['/', 'o', '*']
for bar in plt.bar(lables, values):
    bar.set_hatch(patterns.pop(0))
plt.show()


# In[130]:


lables = ["A", "B", "C"]
values = [1,4 ,2]
bars = plt.bar(lables, values)
patterns = ['/', 'o', '*']
for bar in bars:
    bar.set_hatch(patterns.pop(0))
plt.show()


# In[58]:


get_ipython().run_line_magic('matplotlib', 'inline')
lables = ["A", "B", "C"]
values = [1,4 ,2]
bars = plt.bar(lables, values)
patterns = ['/', 'o', '*']
for bar in bars:
    bar.set_hatch(patterns.pop(0))
plt.show()


# In[52]:


get_ipython().run_line_magic('matplotlib', 'notebook')
lables = ["A", "B", "C"]
values = [1,4 ,2]
bars = plt.bar(lables, values)
patterns = ['/', 'o', '*']
for bar in bars:
    bar.set_hatch(patterns.pop(0))
plt.show()


# In[53]:


get_ipython().run_line_magic('matplotlib', 'notebook')
x = [1,2,4,5,6]
y = [2,4,8,10,12]
x2 = np.arange(0,4, 0.5)



plt.plot(x,y,'r*--' ,label = "2x")
plt.plot(x2, x2**2)

plt.title("this is a good graph")
plt.xlabel("X axis")
plt.ylabel("Y axis")

plt.xticks([1,2,3,4,5,6,7,8,9,10])

plt.legend()
plt.show()


# In[54]:


get_ipython().run_line_magic('matplotlib', 'notebook')
x = [1,2,4,5,6]
y = [2,4,8,10,12]
x2 = np.arange(0,4, 0.5)



plt.plot(x,y,'r*--' ,label = "2x")
plt.plot(x2, x2**2)
plt.grid(True)
plt.title("this is a good graph")
plt.xlabel("X axis")
plt.ylabel("Y axis")

plt.xticks([1,2,3,4,5,6,7,8,9,10])

plt.legend()
plt.show()


# In[55]:


x = [1,2,4,5,6]
y = [2,4,8,10,12]
plt.plot(x,y,color='blue',marker='o',linestyle='',markersize=5)
plt.grid(True)
plt.show()


# In[58]:


plt.plot(x,y,'b<',alpha=.7) # alpha can be specified on a scale 0 to 1
plt.show()


# In[59]:


days=[1,2,3,4,5,6,7]
max_t=[50,51,52,48,47,49,46]
min_t=[43,42,40,44,33,35,37]
avg_t=[45,48,48,46,40,42,40]
plt.plot(days, max_t, label="max")
plt.plot(days, min_t, label="min")
plt.plot(days, avg_t, label="average")
plt.legend()
plt.show()


# In[60]:


# Legend at different location with shadow enabled and fontsize set to large
plt.ioff()
plt.plot(days, max_t, label="max")
plt.plot(days, min_t, label="min")
plt.plot(days, avg_t, label="average")

plt.legend(shadow=True,fontsize='medium')

plt.show()


# In[61]:


company=['GOOGL','AMZN','MSFT','FB']
revenue=[90,136,85,27]
plt.bar(company,revenue, label="Revenue")
plt.ylabel("Revenue(Bln)")
plt.title('US Technology Stocks')
plt.legend()
plt.show()


# In[62]:


company=['GOOGL','AMZN','MSFT','FB']
revenue=[90,136,85,27]
profit=[40,7,34,12]
xpos = np.arange(len(company)) # 0 , 1 , 2, 3

plt.bar(company,revenue, width=0.6, label="Revenue") # x = 0, x = 0.4 
plt.bar(xpos-0.4,profit, width=0.4,label="Profit")# x = -.4  , x = 0
plt.xticks(xpos,company)
plt.ylabel("Revenue(Bln)")
plt.title('US Technology Stocks')
plt.legend()
plt.show()


# In[63]:


exp_vals = [1400,600,300,300,250] #expenses 
exp_labels = ["Home Rent","Food","Phone/Internet Bill","Car","Other Utilities"]
plt.pie(exp_vals,labels=exp_labels)
plt.show()


# In[66]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.axis("equal")
plt.pie(exp_vals,labels=exp_labels, autopct='%0.5f%%',radius=2.4 )
plt.show()


# In[67]:


get_ipython().run_line_magic('matplotlib', 'notebook')
plt.axis("equal")
plt.pie(exp_vals,labels=exp_labels, autopct='%1.2f%%' )
plt.show()


# In[69]:


get_ipython().run_line_magic('matplotlib', 'notebook')
plt.axis("equal")
plt.pie(exp_vals,labels=exp_labels, shadow=True, autopct='%1.4f%%',radius=1.75,explode=[0,0.7,0.1,0,0])
plt.show() # formatting 


# In[70]:


plt.axis("equal")
plt.pie(exp_vals,labels=exp_labels, shadow=True, autopct='%1.1f%%',radius=1.5,explode=[0,0,0,0.1,0.2],startangle=180)
plt.show()


# In[71]:


import numpy as np
import pandas as pd


# In[72]:


gas = pd.read_csv('gas_prices.csv')
gas


# In[73]:


a = gas.Year.values
plt.figure(figsize=(10,5))
plt.plot(gas.Year , gas.USA, label= "USA")
plt.plot(gas.Year , gas.Canada, label= "Canada")
plt.xticks(a)
plt.legend()
plt.show()


# In[74]:


plt.title("Gas prices in USA and Canada")
plt.figure(figsize=(10,5))
plt.xticks(gas.Year)

plt.plot(gas.Year , gas.USA, label= "USA")
plt.plot(gas.Year , gas.Canada , label= "Canada")

plt.legend()
plt.show()


# In[75]:


plt.title("Gas prices in USA and Canada")
plt.figure(figsize=(8,5))
plt.plot(gas.Year , gas["USA"], 'b-' ,label= "USA", marker = "*")
plt.plot(gas.Year , gas["Canada"] , label= "Canada", marker = "o")
plt.plot(gas.Year , gas["South Korea"] , label= "South Korea", marker = ">")

plt.xlabel("Year")
plt.ylabel("US Dollars")
plt.legend()
plt.xticks(gas.Year[::2])
plt.yticks(np.arange(0,7.5, 0.2))
plt.show()


# In[76]:


gas.describe()


# In[77]:


gas.shape


# In[78]:


plt.title("Gas prices in USA and Canada")
plt.figure(figsize=(8,5))

countries = ["USA", "Canada", "Italy", "UK" , "France"]

for i in gas:
    if i in countries:
        plt.plot(gas.Year, gas[i] , label = i)
        
plt.plot(gas.Year , gas["South Korea"] , label= "South Korea", linewidth = 3)

plt.xlabel("Year")
plt.ylabel("US Dollars")
plt.legend()
plt.xticks(gas.Year[::2])
plt.show()


# In[79]:


plt.title("Gas prices in USA and Canada")
plt.figure(figsize=(8,5))

for i in gas.columns:
    if(i == "Year"):
        continue
    else:
        plt.plot(gas.Year, gas[i] , label = i)
        

plt.xlabel("Year")
plt.ylabel("US Dollars")
plt.legend()
plt.xticks(gas.Year[::2])
plt.show()


# ### Fifa

# In[81]:


import pandas as pd 
fifa = pd.read_csv("fifa_data.csv")
fifa


# In[82]:


for i in fifa.Name:
    print(i)


# In[201]:


PlayerNames = fifa["Name"].to_list()
PlayerNames


# In[204]:


fifa.shape


# In[83]:


if "J. Vardy" in fifa.Name.values:
    print("Yes")


# In[85]:


def searchPlayer(playername):
    return playername in fifa.Name.values


# In[87]:


if (searchPlayer("J. Vardy")):
    print("Yes")
else: 
    print("No")


# In[210]:


def searchPlayer(playername):
    if playername in fifa.Name.values:
        return "Yes"
    return "No"


# In[206]:


for i in PlayerNames: 
    if i == 'J. Vardy':
        print ("J. Vardy was in FIFA 19")


# In[98]:


#Histogram
fifa.shape


# In[214]:


get_ipython().run_line_magic('matplotlib', 'notebook')
plt.hist(fifa.Overall)
plt.xticks(np.arange(40, 110, 5))
plt.yticks([0,2, 10,14 , 20, 30, 1000, 2000, 3000, 4000, 4500, 5000, 6000])
plt.show()


# In[88]:


upove90 = fifa["Name"] [ fifa.Overall >= 90 ]
len(upove90)


# In[89]:


upove90


# In[90]:


upove80 = fifa["Name"] [ fifa.Overall >= 80 ]
len(upove80)


# In[220]:


upove80 = fifa["Name"] [ fifa.Overall >= 80 ]
upove80


# In[227]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(30, 15))
players = fifa.Name[:11]
overall = fifa.Overall[:11]
plt.yticks([80,85,90, 95])
plt.bar(players,overall )


# In[229]:


get_ipython().run_line_magic('matplotlib', 'notebook')
bins = [40, 50, 60 , 70 , 80, 90, 100]
plt.hist(fifa.Overall, bins = bins)
plt.show()


# In[91]:


fifa.loc[fifa['Preferred Foot'] == "Left"].count()[0]


# In[232]:


fifa.shape[0]


# In[92]:


fifa.columns


# In[233]:


leftLeg = (fifa.loc[fifa['Preferred Foot'] == "Left"].count()[0]) /  fifa.shape[0] * 100 
leftLeg


# In[93]:


left = fifa.loc[fifa['Preferred Foot'] == "Left"].count()[0]


# In[96]:


get_ipython().run_line_magic('matplotlib', 'notebook')
right = fifa.loc[fifa['Preferred Foot'] == 'Right'].count()[0]
plt.pie([left, right], labels=['left','right'], autopct='%.2f%%')
plt.show()


# In[242]:


fifa.Nationality.value_counts()


# In[97]:


fifa.Nationality
playercountry = ["Argentina", "Portugal",  "Brazil" , "England"]
countrynumbers = []
for i in playercountry:
    countrynumbers.append(fifa.loc[fifa['Nationality'] == i].count()[0])
countrynumbers


# In[98]:


plt.pie(countrynumbers, labels=playercountry)
plt.show()


# In[241]:


plt.bar(playercountry, countrynumbers)
plt.show()


# In[246]:


allcountries = fifa.Nationality.unique()
allcounts = []
for i in allcountries:
    allcounts.append(fifa.loc[fifa['Nationality'] == i].count()[0])
    
plt.pie(allcounts, labels=allcountries, radius=1)
plt.show()


# In[248]:



plt.pie(fifa.Nationality.value_counts(), labels=fifa.Nationality.value_counts().index, radius=1.5)
plt.show()


# In[249]:


plt.bar( fifa.Nationality.value_counts().index, fifa.Nationality.value_counts())
plt.show()


# In[289]:


plt.style.use('ggplot')

maxcountries = fifa.Nationality.value_counts()[:13].to_list()
maxcountriesnames = fifa.Nationality.value_counts().index[:13] 

maxcountriesnames = maxcountriesnames.to_list()
maxcountriesnames.append("Others")
allcountries = sum(maxcountries)

others = sum(list(fifa.Nationality.value_counts())) - allcountries
maxcountries.append(others)

plt.pie(maxcountries, labels=maxcountriesnames, autopct="%0.2f%%")
plt.show()


# In[250]:


fifa.Weight


# In[255]:


fifa.Weight.isnull().values.any()


# In[256]:


fifa.Weight.isnull().values.sum()


# In[257]:


fifa.Weight = [x.strip("lbs") if type (x) == str else x for x in fifa.Weight]
fifa.Weight
fifa.Weight[0]


# In[258]:


plt.style.use('ggplot')
get_ipython().run_line_magic('matplotlib', 'inline')
fifa.Weight = [int(x.strip('lbs')) if type(x)==str else x for x in fifa.Weight]

light = fifa.loc[fifa.Weight < 125].count()[0]
light_medium = fifa[(fifa.Weight >= 125) & (fifa.Weight < 150)].count()[0]
medium = fifa[(fifa.Weight >= 150) & (fifa.Weight < 175)].count()[0]
medium_heavy = fifa[(fifa.Weight >= 175) & (fifa.Weight < 200)].count()[0]
heavy = fifa[fifa.Weight >= 200].count()[0]

weights = [light,light_medium, medium, medium_heavy, heavy]
label = ['under 125', '125-150', '150-175', '175-200', 'over 200']
explode = (.4,.2,0,0,.4)

#plt.title('Weight of Professional Soccer Players (lbs)')

plt.pie(weights, labels=label, explode=explode, pctdistance=0.8,autopct='%.2f %%', radius=2)
plt.tight_layout()
plt.show()


# In[260]:


def checkplayerweight(playername):
    if (searchPlayer(playername)):
        playerweight = int(fifa["Weight"][fifa.Name == playername].strip("lbs"))
        if playerweight < 125:
            print("under25")
    else:
        print("this player is not in FIFA 19 ")


# In[263]:


plt.style.available


# In[266]:


plt.style.use('fivethirtyeight')
plt.figure(figsize=(10,6))
fifa.Weight = [int(x.strip('lbs')) if type(x)==str else x for x in fifa.Weight]

light = fifa.loc[fifa.Weight < 125].count()[0]
light_medium = fifa[(fifa.Weight >= 125) & (fifa.Weight < 150)].count()[0]
medium = fifa[(fifa.Weight >= 150) & (fifa.Weight < 175)].count()[0]
medium_heavy = fifa[(fifa.Weight >= 175) & (fifa.Weight < 200)].count()[0]
heavy = fifa[fifa.Weight >= 200].count()[0]

weights = [light,light_medium, medium, medium_heavy, heavy]
label = ['under 125', '125-150', '150-175', '175-200', 'over 200']
explode = (.4,.2,0,0,.4)

plt.title('Weight of Professional Soccer Players (lbs)')

plt.pie(weights, labels=label, explode=explode, pctdistance=0.8,autopct='%.2f %%')
plt.show()


# In[291]:


plt.style.use('seaborn')

x = [5, 7, 8, 5, 6, 7, 9, 2, 3, 4, 4, 4, 2, 6, 3, 6, 8, 6, 4, 1]
y = [7, 4, 3, 9, 1, 3, 2, 5, 2, 4, 8, 7, 1, 6, 4, 9, 7, 7, 5, 1]
plt.scatter(x,y , s = 50)
plt.show()


# In[144]:


print (plt.style.available)


# In[293]:


plt.xkcd()
plt.style.use('seaborn-deep')

x = [5, 7, 8, 5, 6, 7, 9, 2, 3, 4, 4, 4, 2, 6, 3, 6, 8, 6, 4, 1]
y = [7, 4, 3, 9, 1, 3, 2, 5, 2, 4, 8, 7, 1, 6, 4, 9, 7, 7, 5, 1]
plt.scatter(x,y , s = 100)
plt.grid(True)
plt.xlabel("Hey X")
plt.ylabel("Hey Y")
plt.title("Hey All")
plt.show()


# In[294]:


plt.rcdefaults()


# In[296]:


plt.style.use('seaborn-deep')

x = [5, 7, 8, 5, 6, 7, 9, 2, 3, 4, 4, 4, 2, 6, 3, 6, 8, 6, 4, 1]
y = [7, 4, 3, 9, 1, 3, 2, 5, 2, 4, 8, 7, 1, 6, 4, 9, 7, 7, 5, 1]
plt.scatter(x,y , s = 100)
plt.xlabel("Hey X")
plt.ylabel("Hey Y")
plt.title("Hey All")
plt.show()


# In[299]:


plt.xkcd()
plt.style.use('seaborn-pastel')
plt.figure()


x = [5, 1, 8, 5, 6, 7, 9, 2, 3, 4, 4, 4, 2, 6, 3, 6, 8, 6, 4, 1]
y = [7, 4, 3, 9, 1, 3, 2, 5, 2, 4, 8, 7, 1, 6, 4, 9, 7, 7, 5, 1]


# first subplot 
plt.subplot(2,2,1)
plt.xlabel("Hey X")
plt.ylabel("Hey Y")
plt.scatter(x,y)


# second subplot 
plt.subplot(2,2,2)
plt.xlabel("Hey X")
plt.ylabel("Hey Y")
plt.plot(x,y, "k*")


# Third subplot 
plt.subplot(2,2,3)
plt.xlabel("Hey X")
plt.ylabel("Hey Y")
plt.plot(x,y, "ro")


# forth subplot 
plt.subplot(2,2,4)
plt.xlabel("Hey X")
plt.ylabel("Hey Y")
plt.plot(x,y, 'b>')




plt.show()


# In[301]:


from IPython.display import HTML
url = 'http://jakevdp.github.io/downloads/videos/double_pendulum_xkcd.mp4'
HTML('<video controls alt="animation" src="{0}">'.format(url))


# In[121]:


from IPython.display import YouTubeVideo
YouTubeVideo('hCvWSzJyHPk')


# In[22]:


YouTubeVideo('hCvWSzJyHPk', width=1000, height=570)


# In[303]:


from IPython.display import Math
Math(r'F(k) = \int_{-\infty}^{\infty} f(x) e^{2\pi i k} dx')


# In[304]:


Math(r'F(k) = 2x^4')


# In[305]:


from IPython.display import Image

# by default Image data are embedded
Embed      = Image(    'https://i.pinimg.com/564x/7e/8f/bf/7e8fbf02865d4c04435a5c89b8d727a2.jpg')

# if kwarg `url` is given, the embedding is assumed to be false
SoftLinked = Image(url='https://4.bp.blogspot.com/-hAHPu5tbMaI/WEQ2uDx33-I/AAAAAAAAABM/jLlTq_9F9nk_OkXI5VQK_Ue_7CPhCMc8QCLcB/s1600/David%2BBeckham21.jpg')

# In each case, embed can be specified explicitly with the `embed` kwarg
# ForceEmbed = Image(url='http://scienceview.berkeley.edu/view/images/newview.jpg', embed=True)


# In[306]:


Embed


# In[307]:


type(Embed)


# In[308]:


SoftLinked


# In[125]:


The100 = Image('https://images-na.ssl-images-amazon.com/images/I/91Elu5k9oIL._SL1500_.jpg',)
The100


# In[310]:


from matplotlib.animation import Fu
from itertools import count
import random

index = count()
x = []
y = []

def animate():
    x.append(next(index))
    y.append(random.ranint(0,5))
    plt.cla()
    plt.plot(x,y)

ani = FuncAnimation(plt.gcf(), animate, interval = 500)

plt.show()


# In[61]:





# In[58]:





# In[68]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import animation, rc
from IPython.display import HTML
# First set up the figure, the axis, and the plot element we want to animate
fig, ax = plt.subplots()

ax.set_xlim(( 0, 2))
ax.set_ylim((-2, 2))

line, = ax.plot([], [], lw=2)
def init():
    line.set_data([], [])
    return (line,)
def animate(i):
    x = np.linspace(0, 2, 1000)
    y = np.sin(2 * np.pi * (x - 0.01 * i))
    line.set_data(x, y)
    return (line,)
anim = animation.FuncAnimation(fig, animate, init_func=init,frames=100, interval=20, blit=True)
rc('animation', html='html5')
anim


# In[316]:


import pandas 
pandas.__version__


# In[ ]:




