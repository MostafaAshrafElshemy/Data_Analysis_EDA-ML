#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
print (math)


# In[2]:


signal_power = 50
noise_power = 10
ratio = signal_power / noise_power
decibels = 10 *math.log10(ratio)
redians = 0.7
hieght = math.sin(redians)
print(ratio , decibels , hieght)


# In[3]:


degrees = 45
radians = degrees/ 360.0 * 2 * math.pi
print(radians)


# In[4]:


x = math.sin(degrees / 360 * 2 * math.pi)
y = math.exp(math.log(x+1))
z = x+y
print (z)


# In[5]:


x


# In[6]:


y


# In[7]:


hours = 1
minutes = hours * 60


# In[8]:


minutes


# In[9]:


x = math.sqrt(5)


# In[10]:


x


# In[11]:


math.pi


# In[ ]:





# opertion 

# In[12]:


3 + 4 - 5 * 1 / 6 + 12


# In[13]:


2 ** 5


# In[14]:


1000 ** 3


# In[15]:


x = 2
y = 4
x+y <= 4


# In[16]:


x + y <= 6


#  **Conditions**                            
#  '>'           
#  '<'            
#  '=='            
#  '>='            
#  '<='         
#  '!='            

# In[17]:


x  = 2
y  = 4
x==y


# In[18]:


x != y


# In[19]:


3 > 4 > 5


# In[20]:


3 > 2 > 1


# In[21]:


# Example : X = 3 , Y = 4  IF X > 5 OR Y > 4 print(okay)

x = 3
y = 7
if x > 5 or y > 4 :
    print("okay")


# In[22]:


# Example : X = 3 , Y = 4  IF X > 5 OR Y > 4 print(okay)
x = 2
y = 4
if (x == 4 and y == 4) :
   print("okay")


# In[23]:


# Example : X = 3 , Y = 4  IF X > 5 OR Y > 4 print(okay)
x = 2
y = 4
if (x == 2 and y == 4 and x != y) :
    print("okay")


# In[ ]:





# In[ ]:




