#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import datetime
import os
import time 
import matplotlib.pyplot as plt
import seaborn as sns
import gc
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


sns.set(rc={'figure.figsize' :(12,5)});
plt.figure(figsize=(12,5));


# In[3]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# In[4]:


x =[1,2,3,4,5,6,7,8,9]
y =[3,5,7,2,9,5,3,5,9]
plt.plot(x,y)


# In[ ]:


x =[1,2,3,4,5,6,7,8,9]
y =[3,5,7,2,9,5,3,5,9]

plt.title("this is a linear graph")
plt.plot(x,y)
plt.show()


# In[ ]:


x =[1,2,3,4,5,6,7,8,9]
y =[3,5,7,2,9,5,3,5,9]

plt.title("this is a good graph" , fontdict={'fontname': 'comic sans Ms' , 'fontsize':20})
plt.xlabel("time")
plt.ylabel("price")
plt.plot(x,y)
plt.show()


# In[ ]:


x =[1,2,3,4,5,6,7,8]
y =[2,4,6,8,10,12,14,16]

plt.plot(x,y , label = "y = 2x" , color = "gray", linewidth = 2.5 , marker ="*" , linestyle="--")
plt.title("this is a good graph")
plt.xlabel("x axis")
plt.ylabel("y axix")
plt.legend()
plt.show()


# In[ ]:


x =[1,2,3,4,5,6,7,8]
y =[2,4,6,8,10,12,14,16]

plt.plot(x,y , label = "y = 2x" , color = "gray", linewidth = 3 , marker ="+" , linestyle=":")
plt.title("this is a good graph")
plt.xlabel("x axis")
plt.ylabel("y axix")
plt.legend()
plt.show()


# In[ ]:


x =[1,2,3,4,5,6,7,8]
y =[2,4,6,8,10,12,14,16]

plt.plot(x,y ,'b>--' ,label = "2x" )
plt.title("this is a good graph")
plt.xlabel("x axis")
plt.ylabel("y axix")
plt.legend()
plt.show()


# In[ ]:


x =[1,2,3,4,5,6,7,8]
y =[2,4,6,8,10,12,14,16]

plt.plot(x,y ,'go:' ,label = "2x" )
plt.title("this is a good graph")
plt.xlabel("x axis")
plt.ylabel("y axix")
plt.legend()
plt.show()


# In[ ]:


x =[1,2,3,4,5,6,7,8]
y =[2,4,6,8,10,12,14,16]
x2 = np.arange(0,4.5 , 0.5) # numpy

plt.plot(x,y ,'b>--' ,label = "equation  1" )
plt.plot(x2 , x2**2, 'g>--' , label="equation 2")

plt.title("this is a good graph")
plt.xlabel("x axis")
plt.ylabel("y axix")
plt.legend(loc="upper right")
plt.show()


# In[ ]:


x =[1,2,3,4,5,6,7,8]
y =[2,4,6,8,10,12,14,16]
x2 = [0,0.5,1,1.5,2,2.5,3,3.5,4]

plt.plot(x,y ,'r>--' ,label = "equation  1" )
plt.plot(x2 , [(x*x) for x in x2], 'g>--' , label="equation 2")

plt.title("this is a good graph")
plt.xlabel("x axis")
plt.ylabel("y axix")
plt.legend(loc = "best")# upper right , lower right >equation
plt.show()


# In[ ]:


#%matplotlip inline
x =[1,2,3,4,5,6,7,8]

y =[2,4,6,8,10,12,14,16]
x2 = np.arange(0,4.5 , 0.5) # numpy

plt.plot(x,y ,'r>--' ,label = "2x" )
plt.plot(x2 , x2**2,  label="blue")

plt.title("this is a good graph")
plt.xlabel("x axis")
plt.ylabel("y axix")

plt.xticks(np.arange(0,7))
plt.yticks(np.arange(0,15))

plt.legend()
plt.show()


# In[ ]:


x =[1,2,3,4,5,6,7,8]

y =[2,4,6,8,10,12,14,16]
x2 = np.arange(0,4 , 0.5) # numpy

plt.plot(x,y ,'r*--' ,label = "2x" )
plt.plot(x2 , x2**2,  label="blue")

plt.title("this is a good graph")
plt.xlabel("x axis")
plt.ylabel("y axix")

plt.xticks([0,1,2,2.5,3,4,5,6,7,8])
plt.yticks([i for i in range (0,7)] + [12])

plt.legend()
plt.show()


# In[ ]:


x =[1,2,3,4,5,6,7,8]

y =[2,4,6,8,10,12,14,16]
x2 = np.arange(0,4 , 0.5) # numpy

plt.plot(x,y ,'r*--' ,label = "2x" )
plt.plot(x2 , x2**2,  label="blue")

plt.title("this is a good graph")
plt.xlabel("x axis")
plt.ylabel("y axix")

plt.xticks([0,1,2,2.5,3,4,5,6,7,8])
plt.yticks(list (range (0,15,2)) + [12])

plt.legend()
plt.savefig("random.pdf")# jpg or 
plt.show()


# In[ ]:


x =[1,2,3,4,5,6,7,8]

y =[2,4,6,8,10,12,14,16]
x2 = np.arange(0,4 , 0.5) # numpy

plt.plot(x,y ,'r*--' ,label = "2x" )
plt.plot(x2 , x2**2,  label="blue")

plt.title("this is a good graph")
plt.xlabel("x axis")
plt.ylabel("y axix")

plt.xticks([0,1,2,2.5,3,4,5,6,7,8])
plt.yticks(list (range (0,15,2)) + [12])

plt.legend()
plt.savefig("random2.jpg", dpi = 350)# jpg or 
#plt.show()


# In[ ]:


plt.rcParams["figure.figsize"]=(10,10)


# In[122]:


x = [1,2,3,4,5]
y = [1,4,9,16,25]

plt.title(r"$y = x^2$")

plt.plot(x,y)
plt.show()


# In[123]:


x = [1,2,3,4,5]
y = [1,4,9,16,25]

plt.title(r"$y = x^2$")
plt.plot(x,y,label =r"$y = x^2$" )
plt.legend(bbox_to_anchor=(1.05 , 1))#---

plt.plot(x,y)
plt.show()


# In[124]:


x = [1,2,3,4,5]
y = [1,4,9,16,25]
z=  [1,4,6,8,15]
plt.title(r"$y = x^2$")
plt.plot(x,y,label =r"$y = x^2$" )
plt.plot(x,z,label =r"$y = x^2$" )
plt.legend(ncol=2)#---

plt.plot(x,y)
plt.show()


# In[125]:


x = [1,2,3,4,5]
y = [1,4,9,16,25]
z=  [1,4,6,8,15]
plt.title(r"$y = x^2$")
plt.plot(x,y,label =r"$y = x^2$" , color="green" )
plt.plot(x,z,label =r"$y = x^2$", color ="grey" )
plt.legend(labelcolor = ["black" ,"red"])#---

plt.plot(x,y)
plt.show()


# In[126]:


x = [1,2,3,4,5]
y = [1,4,9,16,25]
z=  [1,4,6,8,15]
plt.title(r"$y = x^2$")
plt.plot(x,y,label =r"$y = x^2$" , color="green" )
plt.plot(x,z,label =r"$y = x^2$", color ="grey" )
plt.legend(labelcolor = ["black" ,"red"] , frameon = False)#---

plt.plot(x,y)
plt.show()


# In[127]:


x = [1,2,3,4,5]
y = [1,4,9,16,25]
z=  [1,4,6,8,15]
plt.title(r"$y = x^2$")
plt.plot(x,y,label =r"$y = x^2$" , color="green" )
plt.plot(x,z,label =r"$y = x^2$", color ="grey" )
plt.legend(labelcolor = ["black" ,"red"] , title = "key map")#---

plt.plot(x,y)
plt.show()


# In[128]:


x = [1,2,3,4,5]
y = [1,4,9,16,25]
z=  [1,4,6,8,15]
plt.title(r"$y = x^2$")
plt.plot(x,y,label =r"$y = x^2$" , color="green" )
plt.plot(x,z,label =r"$y = x^2$", color ="grey" )
plt.legend(labelcolor = ["black" ,"red"])#---

ax=plt.gca()  # get current axis
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#plt.plot(x,y)
plt.show()


# In[129]:


x = [1,2,3,4,5]
y = [1,4,9,16,25]
z=  [1,4,6,8,15]
plt.title(r"$y = x^2$")
plt.plot(x,y,label =r"$y = x^2$" , color="green" )
plt.plot(x,z,label =r"$y = x^2$", color ="grey" )
plt.legend(labelcolor = ["black" ,"red"])#---

ax=plt.gca()  # get current axis
#ax.spines['top'].set_visible(False)
#ax.spines['right'].set_visible(False)
#plt.plot(x,y)
ax.set_frame_on(False)
plt.show()


# In[130]:


x = [1,2,3,4,5]
y = [1,4,9,16,25]
z=  [1,4,6,8,15]
plt.title(r"$y = x^2$")
plt.plot(x,y,label =r"$y = x^2$" , color="green" )
plt.plot(x,z,label =r"$y = x^2$", color ="grey" )
plt.legend(labelcolor = ["black" ,"red"])#---

ax=plt.gca()  # get current axis
#ax.spines['top'].set_visible(False)
#ax.spines['right'].set_visible(False)
#plt.plot(x,y)
ax.set_frame_on(False)
plt.xticks([])
plt.yticks([])
plt.show()


# In[131]:


####  3D PLOTS


# In[132]:


plt


# In[133]:


X=[1,2,3,4,5,6]
Y=[2,4,6,8,10,12]
x2=np.arange(0,4,0.5)
plt.figure(figsize= (8,4 ), dpi = 250)
plt.plot(x,y, 'r*--' , label = "2x")
plt.plot(x2 , x2**2)

plt.title("this is a good graph")
plt.xlabel(" x axis")
plt.ylabel(" y axis")

plt.xticks(np.arange(1,15))

plt.legend()
plt.savefig("goodgraph.jpg")
plt.show()


# In[134]:


X=[1,2,3,4,5,6]
Y=[2,4,6,8,10,12]
x2=np.arange(0,4,0.5)
plt.figure(figsize= (10,4 ), dpi = 250)
plt.plot(x,y, 'r*--' , label = "2x")
plt.plot(x2 , x2**2)

plt.title("this is a good graph")
plt.xlabel(" x axis")
plt.ylabel(" y axis")

plt.xticks(np.arange(1,15))

plt.legend()
plt.savefig("goodgraph.jpg")
plt.show()


# In[135]:


lables = ["A","B","C","D"]
values = [2,5,3,9]
plt.bar(lables,values , color="#8f285b")
plt.yticks(np.arange(0,15,1))
plt.show()


# In[136]:


lables = ["A","B","C","D"]
values = [6,15,13,9]
bars=plt.bar(lables,values , color="#8f285b")
bars[0].set_hatch('/')
bars[1].set_hatch('o')
bars[2].set_hatch('*')
bars[3].set_hatch('/')
plt.yticks(np.arange(0,15,1))
plt.show()


# In[137]:


lables = ["A","B","C"]
values =[10,25,15]
plt.bar(lables , values , color="#034dad")[0].set_hatch("/")
plt.show()


# In[138]:


lables = ["A","B","C"]
values =[10,25,15]
plt.bar(lables , values , color="#034dad")[1].set_hatch("o")
plt.show()


# In[139]:


lables = ["A","B","C"]
values =[10,25,15]
bars=plt.bar(lables , values )
patterns = ['/','o','*']
for bar in bars:
    bar.set_hatch(patterns.pop(0))
plt.show()


# In[140]:


get_ipython().run_line_magic('matplotlib', 'notebook')
lables = ["A","B","C"]
values =[10,25,15]
bars=plt.bar(lables , values )
patterns = ['/','o','*']
for bar in bars:
    bar.set_hatch(patterns.pop(0))
plt.show()


# In[141]:


get_ipython().run_line_magic('matplotlib', 'notebook')
x=[1,2,3,4,5,6]
y=[2,4,6,8,10,12]
x2=np.arange(0,4,0.5)

plt.plot(x,y,'r*--' , label = "2x")
plt.plot(x2, x2**2)

plt.title("this is a good graph")
plt.xlabel("x axis")
plt.ylabel("y axis")

plt.xticks([1,2,3,4,5,6,7,8,9,10,15])

plt.legend()
plt.show()


# In[142]:


x = [1,2,4,5,6]
y = [2,4,8,10,12]
plt.plot(x,y,color='blue',marker='*', linestyle='' , markersize=5)
plt.grid(True)
plt.show()


# In[143]:


x = [1,2,4,5,6]
y = [2,4,8,10,12]
plt.plot(x,y,color='blue',marker='o', linestyle='' , markersize=5)
plt.grid(True)
plt.show()


# In[144]:


plt.plot(x,y,'b<',alpha=.9)
plt.show()


# In[145]:


#legend at different location with shadow enabled and fontsize set to large
days=[1,2,3,4,5,6,7]
max_t=[50,56,58,52,54,35,45]
min_t=[43,45,48,41,39,31,37]
avg_t=[45,48,52,45,44,33,42]
plt.plot(days ,max_t ,label="max")
plt.plot(days ,min_t ,label="min")
plt.plot(days ,avg_t ,label="average")

plt.legend()
plt.show()


# In[146]:


plt.ioff()
#max_t=[50,56,58,52,54,35,45]
#min_t=[43,45,48,41,39,31,37]
#avg_t=[45,48,52,45,44,33,42]
plt.plot(days ,max_t ,label="max")
plt.plot(days ,min_t ,label="min")
plt.plot(days ,avg_t ,label="average")

plt.legend(shadow=True , fontsize='medium')
plt.show()


# In[147]:


company=['GOOGLE','AMZN','MSFT','FB']
revenue=[90 ,140,75,50]
plt.bar(company,revenue , label="Revenue")
plt.ylabel("revenue(Bln)")
plt.title('US Technology Stocks')
plt.legend()
plt.show()


# In[148]:


company=['GOOGLE','AMZN','MSFT','FB']

revenue=[90 ,140,75,50]
profit =[40,60,20,12]
xpos =np.arange(len(company))

plt.bar(company,revenue ,width=0.4 ,label="Revenue")
plt.bar(xpos-0.4,profit ,width=0.4 ,label="profit")
plt.xticks(xpos , company)
plt.ylabel("revenue(Bln)")
plt.title('US Technology Stocks')
plt.legend()
plt.savefig("US Technology.jpg")
plt.show()


# In[149]:


exp_vals=[1400,600,300,500,250] #expenses
exp_labels =["Home Rent","Food","phone/Internet Bill","Car","other Utilities"]
plt.pie(exp_vals , labels=exp_labels)
plt.savefig("US Technology_.jpg")
plt.show()


# In[150]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.axis("equal")
plt.pie(exp_vals , labels = exp_labels , autopct='%0.5f%%' , radius =2.1)
plt.savefig("US Technology_1.jpg")
plt.show()


# In[151]:


get_ipython().run_line_magic('matplotlib', 'notebook')
plt.axis("equal")
plt.pie(exp_vals , labels = exp_labels , autopct='%0.5f%%' , radius =2.1)
plt.show()


# In[152]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.axis("equal")
plt.pie(exp_vals , labels = exp_labels , autopct='%0.5f%%' , radius =2.4,explode=[0,0.5,0.3,0.6,0.2])
plt.savefig("US Technology.jpg")
plt.show()


# In[153]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.axis("equal")
plt.pie(exp_vals , labels = exp_labels , autopct='%0.5f%%' , radius =2.4,explode=[0,0.5,0.3,0.6,0.2] ,startangle=180)
plt.savefig("US Technology__.jpg")
plt.show()


# In[154]:


import numpy as np
import pandas as pd
gas = pd.read_csv('gas_prices.csv')
gas


# In[155]:


a = gas.Year.values
plt.figure(figsize= (10,5 ), dpi = 250)
plt.plot(gas.Year , gas.USA , label ="USA")
plt.plot(gas.Year , gas.Canada , label ="Canada")
plt.xticks(a)
plt.legend()
plt.show()


# In[156]:


#a = gas.Year.values
plt.title("Gas Prices in USA and Canada")
plt.figure(figsize= (10,5 ), dpi = 250)

plt.xticks(gas.Year)

plt.plot(gas.Year , gas.USA , label ="USA")
plt.plot(gas.Year , gas.Canada , label ="Canada")

plt.legend()
plt.show()


# In[157]:


plt.title("Gas Prices in USA and Canada")
plt.figure(figsize= (10,5 ))
plt.plot(gas.Year , gas["USA"] ,'b-', label ="USA" , marker ="*")
plt.plot(gas.Year , gas["Canada"] , label ="Canada" , marker ="o")
plt.plot(gas.Year , gas["South Korea"] , label ="South Korea" , marker=">")

plt.xlabel("Year")
plt.ylabel("US Dollars")
plt.legend()
plt.xticks(gas.Year[::2])
plt.yticks(np.arange(0,7.5,0.2))
plt.show()


# In[158]:


gas.describe()


# In[159]:


gas.shape


# In[160]:


plt.title("Gas prices in USA and Canada")
plt.figure(figsize=(10,5))

conutries =["USA","Canada","Italy","UK","France"]

for i in gas:
    if i in conutries:
        plt.plot(gas.Year , gas[i] , label =i)
        
plt.plot(gas.Year , gas["South Korea"] , label="South Korea" , linewidth = 5)
plt.xlabel("Year")
plt.ylabel("us dollars")
plt.legend()
plt.xticks(gas.Year[::2])
plt.show()


# In[161]:


plt.title("Gas prices in USA and Canada")
plt.figure(figsize=(10,5))

#conutries =["USA","Canada","Italy","UK","France"]

for i in gas.columns:
    if (i=="Year"):
        continue
    else:    
        plt.plot(gas.Year , gas[i] , label =i)
        
#plt.plot(gas.Year , gas["South Korea"] , label="South Korea" , linewidth = 5)
plt.xlabel("Year")
plt.ylabel("us dollars")
plt.legend()
plt.xticks(gas.Year[::2])
plt.show()


# In[162]:


import pandas as pd
fifa =pd.read_csv("fifa_data.csv")
plt.savefig("fifa_data__.jpg")
fifa


# In[163]:


for i in fifa.Name:
    print(i)


# In[164]:


PlayerNames = fifa["Name"].to_list()
PlayerNames


# In[165]:


fifa.shape


# In[166]:


if  "J. Vardy" in fifa.Name.values:
    print("Yes")


# In[167]:


def searchPlayer(playername):
    return playername in fifa.Name.values


# In[168]:


if (searchPlayer("J. Vardy")):
    print("Yes")
else:
    print("No")


# In[169]:


upove90 = fifa["Name"] [fifa.Overall>= 92]
len(upove90)


# In[170]:


upove90


# In[171]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(30 ,15))
Players = fifa.Name[:11]
overall = fifa.Overall[:11]

plt.yticks([80,85,90,95])
plt.bar(Players,overall)


# In[172]:


fifa.loc[fifa['Preferred Foot'] =="Right"].count()[2]


# In[173]:


fifa.shape[0]


# In[174]:


fifa.columns


# In[175]:


leftleg =(fifa.loc[fifa['Preferred Foot'] =="Left"].count()[0]) / fifa.shape[0] *100
leftleg


# In[176]:


Left = fifa.loc[fifa['Preferred Foot'] == "Left"].count()[0]


# In[177]:


right = fifa.loc[fifa['Preferred Foot'] == 'Right'].count()[0]
plt.pie([Left , right],labels=['Left' , 'right'], autopct ='%.2f%%')
plt.show()


# In[178]:


fifa.loc[fifa['Preferred Foot'] =="Right"].count()[2]


# In[179]:


fifa.loc[fifa['Preferred Foot'] =="Left"].count()[2]


# In[180]:


fifa.Nationality.value_counts()


# In[181]:


fifa.Nationality
playercountry =["Argentina" ,"Portugal","Brazil","England"]
countrynumbers =[]
for i in playercountry:
    countrynumbers.append(fifa.loc[fifa['Nationality'] == i].count()[0])
countrynumbers    


# In[182]:


plt.pie(countrynumbers,labels=playercountry)
plt.show()


# In[ ]:





# In[183]:


plt.bar(playercountry , countrynumbers)
plt.show()


# In[184]:


plt.style.use('ggplot')
   
maxcountries = fifa.Nationality.value_counts()[:25].to_list()
maxcountriesnames = fifa.Nationality.value_counts().index[:25]

maxcountriesnames = maxcountriesnames.to_list()
maxcountriesnames.append("Others")
allcountries = sum(maxcountries)

others = sum(list(fifa.Nationality.value_counts())) - allcountries
maxcountries.append(others)


plt.pie(maxcountries , labels=maxcountriesnames ,autopct ="%0.2f%%")
plt.show()


# In[185]:


bins =[40,50,60,70,80,90,100]
plt.hist(fifa.Overall , bins = bins)
plt.show()


# In[186]:


#PLT.XKCD
plt.style.use('seaborn-pastel')
plt.figure()

x  =[5,1,8,5,8,6,5,2,9,4,9,5,7,6,3,1,4,5,6,1]
y = [8,6,5,4,7,9,8,5,2,6,5,4,9,2,3,4,5,8,7,1]
# frist subplot
plt.subplot(2,2,1)
plt.xlabel(" Hey X")
plt.ylabel("Hey Y")
plt.scatter(x,y)


# second subplot
plt.subplot(2,2,2)
plt.xlabel(" Hey X")
plt.ylabel("Hey Y")
plt.plot(x,y ,"k*")


# third subplot
plt.subplot(2,2,3)
plt.xlabel(" Hey X")
plt.ylabel("Hey Y")
plt.plot(x,y ,"ro")


# forth subplot
plt.subplot(2,2,4)
plt.xlabel(" Hey X")
plt.ylabel("Hey Y")
plt.plot(x,y ,"b>")

plt.show()


# In[187]:


#PLT.XKCD
plt.style.use('seaborn-pastel')
plt.figure()

x  =[5,1,8,5,8,6,5,2,9,4,9,5,7,6,3,1,4,5,6,1]
y = [8,6,5,4,7,9,8,5,2,6,5,4,9,2,3,4,5,8,7,1]
# frist subplot
plt.subplot(4,1,1)
plt.xlabel(" Hey X")
plt.ylabel("Hey Y")
plt.scatter(x,y)


# second subplot
plt.subplot(4,1,2)
plt.xlabel(" Hey X")
plt.ylabel("Hey Y")
plt.plot(x,y ,"k*")


# third subplot
plt.subplot(4,1,3)
plt.xlabel(" Hey X")
plt.ylabel("Hey Y")
plt.plot(x,y ,"ro")


# forth subplot
plt.subplot(4,1,4)
plt.xlabel(" Hey X")
plt.ylabel("Hey Y")
plt.plot(x,y ,"b>")

plt.show()


# In[188]:


#PLT.XKCD
plt.style.use('seaborn-pastel')
plt.figure()

x  =[5,1,8,5,8,6,5,2,9,4,9,5,7,6,3,1,4,5,6,1]
y = [8,6,5,4,7,9,8,5,2,6,5,4,9,2,3,4,5,8,7,1]
# frist subplot
plt.subplot(1,4,1)
plt.xlabel(" Hey X")
plt.ylabel("Hey Y")
plt.scatter(x,y)


# second subplot
plt.subplot(1,4,2)
plt.xlabel(" Hey X")
plt.ylabel("Hey Y")
plt.plot(x,y ,"k*")


# third subplot
plt.subplot(1,4,3)
plt.xlabel(" Hey X")
plt.ylabel("Hey Y")
plt.plot(x,y ,"ro")


# forth subplot
plt.subplot(1,4,4)
plt.xlabel(" Hey X")
plt.ylabel("Hey Y")
plt.plot(x,y ,"b>")

plt.show()


# In[189]:


from IPython.display import HTML
url ='http://jakevdp.github.io/downloads/videos/double_pendulum_xkcd.mp4'
HTML('<video controls alt="animation" src="{0}">'.format(url))


# In[190]:


from IPython.display import YouTubeVideo
YouTubeVideo('Vw0vQa7fpuI')


# In[191]:


YouTubeVideo('Vw0vQa7fpuI' , width=1000 , heigh=1000)


# In[192]:


from IPython.display import Math
Math(r'F(k) =\int_{-\infty}^{\infty}  f(x) e^{2\pi  i  k} dx')


# In[193]:


Math(r' F(k) = 2x^4')


# In[194]:


from IPython.display import Image

#by default Image data are embedded

Embedd = Image('https://www.bing.com/images/search?q=Beautiful+Rain+Pics&form=IARSLK&first=1&tsc=ImageHoverTitle.jpg')

softlinked = Image(url='https://www.bing.com/images/search?view=detailV2&ccid=ZmIV38NC&id=DFBECA46135FEBB611B2123940359572D59966BF&thid=OIP.ZmIV38NCfsMgNXPcXzWL2AHaEo&mediaurl=https%3a%2f%2fgeeglenews.com%2fct-geeglenews%2fuploads%2f2016%2f05%2fbeautiful-rain-wallpaper-5.jpg&cdnurl=https%3a%2f%2fth.bing.com%2fth%2fid%2fR.666215dfc3427ec3203573dc5f358bd8%3frik%3dv2aZ1XKVNUA5Eg%26pid%3dImgRaw%26r%3d0&exph=1600&expw=2560&q=Beautiful+Rain+Pics&simid=608032125428264728&FORM=IRPRST&ck=F89F2ADA1ADBA26B07F5B9444930B86D&selectedIndex=0&ajaxhist=0&ajaxserp=0')


# In[195]:


Embedd 


# In[196]:


import librosa
print(librosa.__version__)


# In[ ]:





# In[ ]:





# In[197]:


## Reshape dataframe using stack / unstack


# In[198]:


import pandas as pd
df = pd.read_excel("stocks.xlsx",header=[0,1])
df


# In[199]:


facebook={"Quarter": ["5-jun-17","6-jun-17","7-jun-17","8-jun-17","9-jun-17"], "price":[1,2,3,4,5] ,"price Ratio":[1,2,3,4,5]}
        
fbdf =pd.DataFrame(facebook)
fbdf


# In[200]:


x = df
x = x.stack()

x.to_csv("comapny.csv")


# In[201]:


df.stack(level=0)


# In[202]:


df_stacked = df.stack()
df_stacked


# In[203]:


df_stacked.unstack()


# In[204]:


df3=pd.read_excel("price 25-9-2022 (2) اخر تعديل.xlsx" , header=[0])
df3


# In[205]:


df2.stack(level=0)


# In[ ]:


df2=pd.read_excel("stocks_3_levels.xlsx" , header=[0,1,2])
df2


# In[ ]:


df2.stack()


# In[ ]:


df2.stack(level=0)


# In[ ]:


df2.stack(level=1)


# In[ ]:


### Crosstab Tutorial


# In[ ]:


import pandas as pd
df =pd.read_excel("survey.xls")
df


# In[ ]:


pd.crosstab(df.Nationality , df.Handedness)


# In[ ]:


pd.crosstab(df.Sex , df.Handedness)


# In[ ]:


## Margins


# In[ ]:


pd.crosstab(df.Sex ,df.Handedness , margins =True )


# In[ ]:


pd.crosstab([df.Sex ,df.Nationality],[df.Handedness  ], margins =True )


# In[ ]:


pd.crosstab(df.Sex , [df.Handedness, df.Nationality] , margins =True )


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


normalize


# In[ ]:


pd.crosstab(df.Sex , df.Handedness , normalize ='index')


# In[ ]:


## Aggfunc and Values


# In[ ]:


import numpy as np
pd.crosstab(df.Sex , df.Handedness , values=df.Age , aggfunc=np.average)


# In[ ]:


# SQL DATA BASE
import pandas as pd
import sqlalchemy


# In[ ]:


#engine= sqlalchemy.create_engine('mysql+pymysql://root:@localhost:3306/application')


# In[ ]:


import pandas as pd
population= pd.read_json("worldpopulation.json")
df


# In[ ]:


population["country"][population.population == population.population.max()][0]


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




