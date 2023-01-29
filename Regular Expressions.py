#!/usr/bin/env python
# coding: utf-8

# ### Today's Content

# #### Regular Expressions 
# #### Getting started with kaggle
# #### More questions and answers

# In[92]:


def displayname(name):
    print (name)


# In[95]:


class Human:
    def __init__(this, name, passw, humanage):
        
        #attributes
        this.username = name
        this.__password = passw
        this.age = humanage
        this.__creditCard = 111111
    
    def printName(this):
        print ("my name is: ", this.username)
    
    def SetPassword(self, oldpassword, newpassword):
        if (oldpassword == self.__password):
            self.__password = newpassword
    
    def GetPassword(self):
        print(self.__password)
    
    def addnewfunction(self):
        displayname(self.username)
        


# In[183]:


Ahmed= Human("Ahmed", 123, 23)
Ahmed.addnewfunction()


# In[168]:


Ahmed.printName()


# In[104]:


class Employee(Human):
    def __init__(self): #Constructor 
        print ("Hi, you're so bad , so you take 1500LE. ")
        Human.__init__(self, "Ahmed", 123, 245)
    
    
    def printprive(self):
        Human.__privateFunction()


# In[105]:


employee1 = Employee()
employee1.age


# In[106]:


employee1.printName()


# In[94]:


a.email


# In[72]:


a.email


# In[162]:


class Helper:
    def __init__(self):
        self.name = "Ahmed"
        self.age = 23
    def speak(self):
        print("I'm a human and I'm talking")
#task 1 : search for : how to make our class abstract


# In[165]:


class Human(Helper):
    def __init__(self):
        Helper.__init__(self)
        self.gender = "pure evil"
    
    def printage(self):
        print(self.age)        


# In[164]:


class Employee(Helper):
    def __init__(self):
        Helper.__init__(self)
        
class Player(Helper):
    def __init__(self):
        Helper.__init__(self)


# In[161]:





# In[187]:


import Ramadan


# In[188]:


Ramadan.x


# In[189]:


Ramadan.RamdanSession()


# In[193]:


from Ramadan import * # all 


# In[196]:


Person = Mazen()


# In[ ]:





# In[ ]:





# In[94]:


newobject = Human("Ahmed", 23, "ahmed@gmail.com")
newobject2 = Human("Ahmed", 22, "ahmed@gmail.com")
newobject.name


# In[90]:


newobject.creatpassword()


# In[85]:


newobject.showcredit("1231")


# In[143]:


def sum4(x):
    return (x+4)


# In[158]:


class Employee(Human):
    
    def __init__(self): # concustructor override 
        print("hello")
        Human.__init__(self, "Ahmed", 23, "a@a")
        
    def printName(self):
        print("this is ", self.name) #override ( polymorphism )
    
    #Encapsulation 
    
    def setEmpSalary(self):
        self.__empSalary = 2000
     
    def getEmpSalary(self):
        print (self.__empSalary)
    
    
    #Abstraction: #make it eaiser for access functions and data in class 
    
    def __printFname(self):
        print ("first name is :" , "Ahmed")
    
    def __printSname(self):
        print ("Second name is :" , "Hafez")
    
    def __printTname(self):
        print ("Third name is : ", "Ahmed")
        
    def printAll(self):
        self.__printFname()
        self.__printSname()
        self.__printTname()
    
    
    def addvalue(self):
        print (sum4(5))
    


# In[159]:


single , Single, Signal , Divorced, divorced, مطلق , ارملة
gender : 0 , 1 
additional data: 


# In[154]:


#Taks : how to make function overloading in class in python 


# In[ ]:


AI : 
    80% Data
    20% Model (Actual AI)

80% Data:
    90% : Data Cleaning, Filtering
    10% : Apply Data Analysis Algorithm (i.e. Understand Data )
    


# In[197]:


# Regular Expressions ? 

Data preprocessing 
Temprature : kelvin 


# In[2]:


import re # ( Regular expression )


# In[3]:


text_to_search = '''abcdefghijklmnopqurtuvwxyz
ABCDEFGHIJKLMNOPQRSTUVWXYZ
123456789010
Ha HaHa
MetaCharacters (Need to be escaped):
. ^ $ * + ? { } [ ] \ | ( )
abc
cat
mat 
hat
bat
fat
sat
01092329340
coreyms-com
321-555-4321
123.555.1234
123*555*1234
800-555-1234
900-555-1234
Mr. Schafer
Mr Smith
Ms Davis
Mrs. Robinson
Mr. T
'''

sentence = 'Start a sentence and then bring it to an end'


# ### Rules:
# 
# .       - Any Character Except New Line <br>
# \d      - Digit (0-9)<br>
# \D      - Not a Digit (0-9)<br>
# \w      - Word Character (a-z, A-Z, 0-9, _) <br>
# \W      - Not a Word Character<br>
# \s      - Whitespace (space, tab, newline) <br>
# \S      - Not Whitespace (space, tab, newline)<br>
# 
# \b      - Word Boundary <br>
# \B      - Not a Word Boundary <br>
# ^       - Beginning of a String <br>
# $       - End of a String <br>
# 
# []      - Matches Characters in brackets <br>
# [^ ]    - Matches Characters NOT in brackets <br>
# |       - Either Or <br>
# ( )     - Group <br>
# 
# Quantifiers:
# <br>
#  {*}       - 0 or More <br>
#  {+}       - 1 or More <br>
#  ?       - 0 or One <br>
# {3}     - Exact Number <br>
# {3,4}   - Range of Numbers (Minimum, Maximum) <br>
# 
# 
# #### Sample Regexs ####
# 
# [a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+
# 

# In[10]:


print ("\tTab")
print ("tab")
print (r"\tTab") # raw print


# In[12]:


print("Ahmed\nAhmed")
print("----------------")
print(r"Ahmed\nAhmed")


# In[33]:


string = "ahmed"
string.find("med")


# In[34]:


string = "ahm5d"
string.find("med")


# In[27]:


import re
pattern = re.compile(r"abc") # search for abc 
matches = pattern.finditer(text_to_search) # store all the "abc" you found (could be a list )
counter = 0
for elment in matches:
    print(elment.span()[0] , elment.span()[1] )
    counter+=1
print("number of times: ",str(counter))


# In[28]:


pattern = re.compile(r"bca")
matches = pattern.finditer(text_to_search)
for match in matches:
    print(match)


# In[29]:


pattern = re.compile(r".") #matches everything ( excpet new lines )
matches = pattern.finditer(text_to_search)
for match in matches:
    print(match)


# In[35]:


pattern = re.compile(r"\.") #matches only dots 
matches = pattern.finditer(text_to_search)

for match in matches:
    print("start index:", match.span()[0] , "and end index is: ", (match.span()[1])-1) 


# In[37]:


pattern = re.compile(r"coreyms.com") #matches only dots 
matches = pattern.finditer(text_to_search)
l = [ ]
for match in matches:
    print(match)


# In[38]:


pattern = re.compile(r"coreyms\.com") #matches only dots 
matches = pattern.finditer(text_to_search)
for match in matches:
    print(match)


# In[40]:


pattern = re.compile(r"d") #matches only digits  
matches = pattern.finditer(text_to_search)
for match in matches:
    print(match)


# In[4]:


pattern = re.compile(r"\D{3}") #matches only digits  
matches = pattern.finditer(text_to_search)
for match in matches:
    print(match)


# In[41]:


pattern = re.compile(r"\d") #matches only digits  
matches = pattern.finditer(text_to_search)
for match in matches:
    print(match)


# In[42]:


pattern = re.compile(r"\d\d") #matches only two digits  
matches = pattern.finditer(text_to_search)
for match in matches:
    print(match)


# In[25]:


pattern = re.compile(r"\D") #matches only not a digit  
matches = pattern.finditer(text_to_search)
for match in matches:
    print(match)


# In[4]:


pattern = re.compile(r"\w") #matches only lower, uppercase, digits, underscore  
matches = pattern.finditer(text_to_search)
for match in matches:
    print(match)


# In[9]:



pattern = re.compile(r"end$")   # if the word (end) is at the end of the sentence ( string ) 
matches = pattern.finditer(sentence)

for match in matches:
    print(match)

def getIterLength(iterator):
    temp = list(iterator)
    result = len(temp)
    return result

if  (getIterLength(matches) == 0) :
    print ("wrong email address")
else:
    print ('zy el fol')


# In[40]:


#match line numbers 
pattern = re.compile(r"\d\d\d")
matches = pattern.finditer(text_to_search)
for match in matches:
    print(match)


# In[10]:


pattern = re.compile(r"\d\d\d.\d\d\d.\d\d\d\d")
matches = pattern.finditer(text_to_search)
for match in matches:
    print(match)
    


# In[11]:


#character set 
pattern = re.compile(r"\d\d\d[-.]\d\d\d[-.]\d\d\d\d")
matches = pattern.finditer(text_to_search)
for match in matches:
    print(match)


# In[12]:


#character set 
pattern = re.compile(r"[89]00[-.]\d\d\d[-.]\d\d\d\d")
matches = pattern.finditer(text_to_search)
for match in matches:
    print(match)


# In[13]:


#character set 
pattern = re.compile(r"\d\d\d[-.*]555[-.*]\d\d\d\d")
matches = pattern.finditer(text_to_search)
for match in matches:
    print(match)


# In[14]:


#character set 
pattern = re.compile(r"0109")
matches = pattern.finditer(text_to_search)
for match in matches:
    print(match)


# In[15]:


#character set 
pattern = re.compile(r"[1-5]")
matches = pattern.finditer(text_to_search)
for match in matches:
    print(match)


# In[16]:


#character set 
pattern = re.compile(r"[-15]")
matches = pattern.finditer(text_to_search)
for match in matches:
    print(match)


# In[17]:


#character set 
pattern = re.compile(r"[1-5-]")
matches = pattern.finditer(text_to_search)
for match in matches:
    print(match)


# In[19]:


#character set 
pattern = re.compile(r"[az-]")
matches = pattern.finditer(text_to_search)
for match in matches:
    print(match)


# In[18]:


#character set 
pattern = re.compile(r"[a-z]")
matches = pattern.finditer(text_to_search)
for match in matches:
    print(match)


# In[20]:


#character set 
pattern = re.compile(r"[a-zA-Z]")
matches = pattern.finditer(text_to_search)
for match in matches:
    print(match)


# In[21]:


#character set 
pattern = re.compile(r"[^a-zA-Z]") # not a in a character set
matches = pattern.finditer(text_to_search)
for match in matches:
    print(match)


# In[22]:


#character set 
pattern = re.compile(r"[^c]at")
matches = pattern.finditer(text_to_search)
for match in matches:
    print(match)


# In[23]:


#character set ( Quantifier )
pattern = re.compile(r"\d{3}-\d{3}-\d{4}")
matches = pattern.finditer(text_to_search)
for match in matches:
    print(match)


# In[25]:


#character set ( Quantifier )
pattern = re.compile(r"Mr.")
matches = pattern.finditer(text_to_search)
for match in matches:
    print(match)


# In[24]:


#character set ( Quantifier )
pattern = re.compile(r"Mr\.")
matches = pattern.finditer(text_to_search)
for match in matches:
    print(match)


# In[26]:


#character set ( Quantifier )
pattern = re.compile(r"Mr\.?")  # one or zero 
matches = pattern.finditer(text_to_search)
for match in matches:
    print(match)


# In[29]:


#character set ( Quantifier )
pattern = re.compile(r"Mr\.?\s[A-Z]?\w{3}")
matches = pattern.finditer(text_to_search)
for match in matches:
    print(match)
# Mr[.]" "[A-Z]? (upper or lower or digits or underscore) (+ means : at least one )


# In[30]:


#character set ( Quantifier )
pattern = re.compile(r"Mr\.?\s[A-Z]?\w+")
matches = pattern.finditer(text_to_search)
for match in matches:
    print(match)
# Mr[.]" "[A-Z]? (upper or lower or digits or underscore) (+ means : at least one )


# In[31]:


#character set ( Quantifier )
pattern = re.compile(r"M(r|s|rs)\.?\s[A-Z]?\w+")
matches = pattern.finditer(text_to_search)
for match in matches:
    print(match)


# In[32]:


emails = '''
CoreyMSchafer@gmail.com
corey.schafer@university.edu
corey-321-schafer@my-work.net
'''

pattern = re.compile(r'[a-zA-Z]+@[a-zA-Z]+\.com')

matches = pattern.finditer(emails)

for match in matches:
    print(match)


# In[34]:


emails = '''
CoreyMSchafer@gmail.com
corey.schafer@university.edu
corey-321-schafer@my-work.net
'''

pattern = re.compile(r'[a-zA-Z.-]+@[a-zA-Z-]+\.(com|edu|net)')

matches = pattern.finditer(emails)

for match in matches:
    print(match)


# In[36]:


emails = '''
CoreyMSchafer@gmail.com
corey.schafer@university.edu
corey-321-schafer@my-work.net
ahmed@gmail.mywork
'''

pattern = re.compile(r'[a-zA-Z0-9-.]+@[a-zA-Z-]+\.\w{3}')

matches = pattern.finditer(emails)

for match in matches:
    print(match)


# In[ ]:





# In[ ]:





# In[ ]:





# In[37]:


emails = '''
CoreyMSchafer@gmail.com
corey.schafer@university.edu
corey-321-schafer@my-work.net
'''

pattern = re.compile(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+')

matches = pattern.finditer(emails)

for match in matches:
    print(match)


# In[ ]:





# In[38]:


sentence = '1234'
pattern = re.compile(r'\d\d\d')

matches = pattern.finditer(sentence)

for match in matches:
    print(match)


# In[ ]:





# In[ ]:





# In[ ]:





# In[160]:


type(text_to_search)


# In[239]:


"Ahmed"


# In[240]:


'Ahmed'


# In[241]:


'''Ahmed'''


# In[242]:


# Ahmed 


# 

# In[11]:


from Instant import NewPrint


# In[10]:


st.NewPrint(4)


# In[70]:



    


# In[71]:





# In[72]:





# In[20]:





# In[26]:


Ali = Human("Ali")


# In[168]:


class Human:
    
    def __init__(self):
        self.__name = "Ahmed"
    
    def getname(self):
        print (self.__name)
    
    def setname (self, newname ):
        self.__name = newname


# In[169]:


class Employee(Human):
    def __init__(self):
        Human.__init__(self)
    
    
    def welcome(self):
        print ("Welcome emp ")
    


# In[170]:


e1 = Employee()


# In[171]:


e1.name


# minimum requirments for AI Device: 
# 1. Graphics Card : Navidia
# 2. RAM: 16GB
# 3. processor :  >= 9th gen ( intel ) 
# 4. Nividia : RTX2060 ( 6GB ) or bigger 

# Todays Tasks: 
# 
# 1. make sure that we created : linkedin , colab , kaggle, github , twitter account 
# 2. send CV , if not create then send it
# 
# sesstion related : 
# 1. answer notebook , upload it to github 

# **web scrapping**

# In[ ]:




