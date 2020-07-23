#!/usr/bin/env python
# coding: utf-8

# # Minist Dataset : Visualisation using PCA and TSNET

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# In[2]:


minist_train=pd.read_csv("D:\\train_MNIST.csv")


# In[3]:


minist_train.head()


# In[4]:


l=minist_train["label"]


# In[5]:


minist_train.drop("label",axis=1,inplace=True)


# In[6]:


mth=minist_train.head(42000)


# In[7]:


plt.figure(figsize=(7,7))
x=10
n=minist_train.iloc[x].to_numpy()
m=np.asmatrix(n).reshape(28,28)
plt.imshow(m,interpolation="bicubic")


# # PCA Without sklearn
# 

# In[8]:


from sklearn.preprocessing import StandardScaler
ss=StandardScaler()


# In[9]:


ss=ss.fit_transform(mth)
ss


# In[10]:


sst=ss.transpose()
ss.shape


# In[11]:


variance_matrix=np.matmul(sst,ss)


# In[12]:


from scipy.linalg import eigh


# In[13]:


values,vector=eigh(variance_matrix,eigvals=(782,783))


# In[14]:


values.sum()/29735999.999999978


# In[15]:


vector.shape


# In[16]:


vector=vector.transpose()
new_points=np.dot(vector,sst)


# In[17]:


new_points=new_points.transpose()


# In[18]:


new_points=pd.DataFrame(new_points)


# In[19]:


new_points


# In[20]:


direction=pd.concat([new_points,l],axis=1,ignore_index=True)


# In[21]:


direction.columns=["u1","u2","Label"]


# In[22]:


direction["Label"].unique()


# In[23]:


sns.scatterplot("u1","u2",hue=direction["Label"],data=direction)
plt.legend()


# In[24]:


sns.FacetGrid(direction,hue="Label",size=10).map(plt.scatter,"u1","u2")
plt.legend()


# # using pca
# 

# In[25]:


from sklearn.decomposition import PCA


# In[26]:


pca=PCA(n_components=2)


# In[27]:


data=pd.DataFrame(pca.fit_transform(ss),columns=["u1","u2"])


# In[28]:


data


# In[29]:


data1=pd.concat([data,l],ignore_index=True,axis=1)


# In[30]:


data1.columns=["u1","u2","Labels"]


# In[31]:


sns.FacetGrid(data1,hue="Labels",size=10).map(plt.scatter,"u1","u2")
plt.legend()


# In[32]:


pca=PCA(n_components=784)
pca_data=pca.fit_transform(ss)
percentage=pca.explained_variance_/np.sum(pca.explained_variance_)


# In[33]:


cumsum=np.cumsum(percentage)
plt.figure(1,figsize=(10,10))
plt.plot(cumsum*100)
plt.grid()
plt.xlabel("n_components")
plt.ylabel("sum of variance")
plt.show()


# # T-SNET

# In[34]:


from sklearn.manifold import TSNE
tsne=TSNE(n_components=2,random_state=0)


# In[35]:


ssa=tsne.fit_transform(ss[:1000])
data=pd.DataFrame(ssa,columns=["d1","d2"])
data


# In[36]:


data2=pd.concat([data,l[:1000]],ignore_index=True,axis=1)
data2


# In[37]:


data2.columns=["d1","d2","label"]
data2


# In[38]:


sns.FacetGrid(data2,hue="label",size=10).map(plt.scatter,"d1","d2")


# In[39]:


tsne


# In[40]:


tsne=TSNE(n_components=2,random_state=0,perplexity=50,n_iter=5000)


# In[41]:


ssa=tsne.fit_transform(ss[:5000])
data4=pd.DataFrame(ssa,columns=["d1","d2"])
data4


# In[42]:


data5=pd.concat([data4,l[:1000]],ignore_index=True,axis=1)
data5
data5.columns=["d1","d2","label"]
data5
sns.FacetGrid(data2,hue="label",size=10).map(plt.scatter,"d1","d2")


# In[43]:


tsne=TSNE(n_components=2,random_state=0,perplexity=30,n_iter=500)


# In[44]:


ssa=tsne.fit_transform(ss[:])
data4=pd.DataFrame(ssa,columns=["d1","d2"])
data4


# In[45]:


data5


# In[46]:


data5=pd.concat([data4,l],ignore_index=True,axis=1)
data5
data5.columns=["d1","d2","label"]
data5
sns.FacetGrid(data5,hue="label",size=30).map(plt.scatter,"d1","d2")
plt.legend()

