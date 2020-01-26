#Importing the required packages
from flask import Flask, render_template, request
import os
import pandas as pd
from pandas import ExcelFile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import *
from sklearn.metrics import *
from sklearn.model_selection import cross_val_score
import itertools 
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

app = Flask(__name__)

#Routing to initial home page
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/admin_login')
def admin_login():
    return render_template('admin_login.html')
    
@app.route('/admin', methods=['GET','POST'])
def admin():
    user=request.form['un']
    pas=request.form['pw']
    cr=pd.read_excel('admin_cred.xlsx')
    un=np.asarray(cr['Username']).tolist()
    pw=np.asarray(cr['Password']).tolist()
    cred = dict(zip(un, pw))
    if user in un:
        if(cred[user]==pas):
            return render_template('admin.html')
        else:
            k=1
            return render_template('admin_login.html',k=k)
        
    else:
        k=1
        return render_template('admin_login.html',k=k)

@app.route('/admin_printed', methods=['GET','POST'])
def admin_printed():
    trainfile=request.files['admin_doc']
    t=pd.read_excel(trainfile)
    t.to_excel('trainfile.xlsx')
    return render_template('admin_printed.html')


@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/index', methods=['GET','POST'])
def index():
    user=request.form['un']
    pas=request.form['pw']
    cr=pd.read_excel('cred.xlsx')
    un=np.asarray(cr['Username']).tolist()
    pw=np.asarray(cr['Password']).tolist()
    cred = dict(zip(un, pw))
    if user in un:
        if(cred[user]==pas):
            return render_template('index.html')
        else:
            k=1
            return render_template('login.html',k=k)
        
    else:
        k=1
        return render_template('login.html',k=k)

#Routing to page when File Upload is selected 
@app.route('/file_upload')
def file_upload():
    return render_template("file_upload.html")

@app.route('/upload_printed', methods=['GET','POST'])
def upload_printed():
    abc=request.files['printed_doc']

    test1=pd.read_excel(abc)
    test=test1
    train=pd.read_excel('trainfile.xlsx')

    train['TenurePerJob']=0
    for i in range(0,len(train)):
        if train.loc[i,'NumCompaniesWorked']>0:
            train.loc[i,'TenurePerJob']=train.loc[i,'TotalWorkingYears']/train.loc[i,'NumCompaniesWorked']
    a=np.median(train['MonthlyIncome'])
    train['CompRatioOverall']=train['MonthlyIncome']/a        
    full_col_names=train.columns.tolist()
    num_col_names=train.select_dtypes(include=[np.int64,np.float64]).columns.tolist()
    num_cat_col_names=['Education','JobInvolvement','JobLevel','StockOptionLevel']
    target=['Attrition']
    num_col_names=list(set(num_col_names)-set(num_cat_col_names))
    cat_col_names=list(set(full_col_names)-set(num_col_names)-set(target))
    #print("total no of numerical features:",len(num_col_names))
    #print("total no of categorical & ordered features:",len(cat_col_names))
    cat_train=train[cat_col_names]
    num_train=train[num_col_names]

    for col in num_col_names:
        if num_train[col].skew()>0.80:
            num_train[col]=np.log1p(num_train[col])
        
    for col in cat_col_names:
        col_dummies=pd.get_dummies(cat_train[col],prefix=col)
        cat_train=pd.concat([cat_train,col_dummies],axis=1)
    
    Attrition={'Yes':1,'No':0}
    train.Attrition=[Attrition[item] for item in train.Attrition]

    cat_train.drop(cat_col_names,axis=1,inplace=True)

    final_train=pd.concat([num_train,cat_train],axis=1)

    final_train['pr_mean_psh'] = final_train['PerformanceRating'].add(final_train['PercentSalaryHike'])
    final_train['pr_mean_psh']=final_train['pr_mean_psh']/2
    final_train.drop(labels=['PerformanceRating','PercentSalaryHike'],axis=1,inplace=True)
    
    df1=final_train
    for col in list(df1):
        df1[col]=df1[col]/df1[col].max()

    empnum=test['EmployeeNumber']
    test['TenurePerJob']=0
    for i in range(0,len(test)):
        if test.loc[i,'NumCompaniesWorked']>0:
            test.loc[i,'TenurePerJob']=test.loc[i,'TotalWorkingYears']/test.loc[i,'NumCompaniesWorked']
    a=np.median(test['MonthlyIncome'])
    test['CompRatioOverall']=test['MonthlyIncome']/a
    test.drop(labels=['EmployeeNumber'],axis=1,inplace=True)    
    #test.drop(labels=['EmployeeCount','EmployeeNumber','Over18','StandardHours'],axis=1,inplace=True)

    full_col_names=test.columns.tolist()
    num_col_names=test.select_dtypes(include=[np.int64,np.float64]).columns.tolist()
    num_cat_col_names=['Education','JobInvolvement','JobLevel','StockOptionLevel']
    target=['Attrition']
    num_col_names=list(set(num_col_names)-set(num_cat_col_names))
    cat_col_names=list(set(full_col_names)-set(num_col_names)-set(target))
    #print("total no of numerical features:",len(num_col_names))
    #print("total no of categorical & ordered features:",len(cat_col_names))
    cat_test=test[cat_col_names]
    num_test=test[num_col_names]

    for col in num_col_names:
        if num_test[col].skew()>0.80:
            num_test[col]=np.log1p(num_test[col])
        
    for col in cat_col_names:
        col_dummies=pd.get_dummies(cat_test[col],prefix=col)
        cat_test=pd.concat([cat_test,col_dummies],axis=1)


    cat_test.drop(cat_col_names,axis=1,inplace=True)

    final_test=pd.concat([num_test,cat_test],axis=1)

    final_test['pr_mean_psh'] = final_test['PerformanceRating'].add(final_test['PercentSalaryHike'])
    final_test['pr_mean_psh']=final_test['pr_mean_psh']/2
    final_test.drop(labels=['PerformanceRating','PercentSalaryHike'],axis=1,inplace=True)
    #final_test.drop(labels=['HourlyRate','MonthlyRate','DailyRate'],axis=1,inplace=True)
    #final_test.drop(labels=['Gender_Male','Gender_Female'],axis=1,inplace=True)
    #final_test.drop(labels=['Department_Human Resources','Department_Research & Development','Department_Sales',],axis=1,inplace=True)
    #final_test.drop(labels=['WorkLifeBalance_1','WorkLifeBalance_2','WorkLifeBalance_3','WorkLifeBalance_4','RelationshipSatisfaction_1','RelationshipSatisfaction_2','RelationshipSatisfaction_3','RelationshipSatisfaction_4','JobSatisfaction_1','JobSatisfaction_2','JobSatisfaction_3','JobSatisfaction_4','EnvironmentSatisfaction_1','EnvironmentSatisfaction_2','EnvironmentSatisfaction_3','EnvironmentSatisfaction_4'],axis=1,inplace=True)

    df2=final_test
    for col in list(df2):
        df2[col]=df2[col]/df2[col].max()
    
    #list(df2)
    df3=df1[list(df2)]
    #if(list(df3)==list(df2)):
        #print('y')
    #print(list(df2))
    
    X_train=np.asarray(df3)
    Y_train=np.asarray(train['Attrition'])
    X_test=np.asarray(df2)

    test1['EmployeeNumber']=np.asarray(empnum).tolist()
    

    lr=LogisticRegression(solver='liblinear').fit(X_train,Y_train)
    yhat=lr.predict(X_test)
    yhat.tolist()
    test1['Attrition'] = yhat    
    Attrition={1:'Yes',0:'No'}
    test1.Attrition=[Attrition[item] for item in test1.Attrition]
    
    conf=[]
    for i in (lr.predict_proba(X_test).tolist()):
        i= max(i)
        conf.append(i)
    #print(len(conf))
    for j in range(len(conf)):
        conf[j]=conf[j]*100
        conf[j] = round(conf[j], 2)
    test1['Reliability Percentage'] = conf
    
    #added affecting parameters here
    
    l=np.abs(lr.coef_).tolist()
    coefs = [item for sublist in l for item in sublist]
    
    data=np.asarray(df2).tolist()
    
    weights=[]
    for row in data:
        c=np.multiply(row,coefs).tolist()
        weights.append(c)
    
    cols=list(df2)
    L=[]
    for val in weights:
        dic = dict(enumerate(val))
        L.append(dic)

    ColWeights=[]
    for dic in L:
        i=0
        tempDic={}
        for key,value in dic.items():
            key=cols[i]
            tempDic[key]=value
            i=i+1
        ColWeights.append(tempDic)
        
    df_yes=test1[test1.Attrition =='Yes']
    df_no=test1[test1.Attrition =='No']
    
        
    for index, row in df_yes.iterrows():
    
        if(row['Attrition']=='Yes'):
        
            yes_changable_cols=['YearsWithCurrManager',
                                'MonthlyIncome',
                                'YearsInCurrentRole',
                                'DistanceFromHome',
                                'YearsSinceLastPromotion',
                                'JobLevel_1',
                                'JobLevel_2',
                                'JobLevel_3',
                                'JobLevel_4',
                                'BusinessTravel_Non-Travel',
                                'BusinessTravel_Travel_Frequently',
                                'BusinessTravel_Travel_Rarely',
                                'OverTime_Yes']
                       
            Col_Weights_Yes=[]
            for dic in ColWeights:
                a={}
                for k,v in dic.items():
                    if k in yes_changable_cols :
                        a[k]=v
                Col_Weights_Yes.append(a)    
            
        
            AscendingCols=[]
            for dic in Col_Weights_Yes:
                AscendingCols.append((sorted(dic, key=dic.get)))
        
            AllParams=[]
            for h in AscendingCols:
                params=[ h[12], h[11], h[10], h[9], h[8] ]
                AllParams.append(params)
                
            frame=pd.DataFrame(AllParams)
            frame.columns =['YesParam_1','YesParam_2','YesParam_3','YesParam_4','YesParam_5']
    
    df_yes=pd.concat([df_yes, frame], axis=1)
    df_yes = df_yes[np.isfinite(df_yes['Age'])]
    #df_yes=df_yes[df_yes.Age != float('nan')]
    #disp=df_yes[df_yes.Attrition=='Yes']
    disp=df_yes[['EmployeeNumber','Reliability Percentage','YesParam_1','YesParam_2']]
    disp.drop(labels=[],axis=1,inplace=True)
            #print(disp.shape)
    for index, row in df_no.iterrows():
    
        if(row['Attrition']=='No'):
            aff_params_no=['YearsWithCurrManager',
                           'YearsInCurrentRole',
                           'MonthlyIncome',
                           'YearsAtCompany',
                           'TotalWorkingYears']
    
            #MAIN PARAMS FOR NO
            Col_Weights_No=[]
            for dic in ColWeights:
                b={}
                for k,v in dic.items():
                    if k in aff_params_no :
                        b[k]=v
                Col_Weights_No.append(b)    
            
        
            AscendingCols1=[]
            for dic in Col_Weights_No:
                AscendingCols1.append((sorted(dic, key=dic.get)))
        
            AllParams1=[]
            for h in AscendingCols1:
                params1=[ h[4], h[3], h[2], h[1], h[0] ]
                AllParams1.append(params1)
                
            frame1=pd.DataFrame(AllParams1)
            frame1.columns =['NoParam_1','NoParam_2','NoParam_3','NoParam_4','NoParam_5']
    
    df_no=pd.concat([df_no, frame1], axis=1)
    df_no = df_no[np.isfinite(df_no['Age'])]
    #df_no=df_no[df_no.Age !=float('nan')]
    
            #disp=test1[test1.Attrition=='Yes']
            #disp=disp[['EmployeeNumber','Reliability Percentage','AffectingParam_1','AffectingParam_2']]
            #disp.drop(labels=[],axis=1,inplace=True)
            #print(disp.shape)
    
    #for index, row in test1.iterrows():
    
        #if(row['Attrition']=='Yes'):
            #test1['NoParam_1']=' '
            #test1['NoParam_2']=' '
            #test1['NoParam_3']=' '
            #test1['NoParam_4']=' '
            #test1['NoParam_5']=' '
    
        #elif(row['Attrition']=='No'):
            #test1['YesParam_1']=' '
            #test1['YesParam_2']=' '
            #test1['YesParam_3']=' '
            #test1['YesParam_4']=' '
            #test1['YesParam_5']=' '
     
        
    writer = pd.ExcelWriter('Result.xlsx', engine='xlsxwriter')

    #store your dataframes in a  dict, where the key is the sheet name you want
    frames = {'Yes_Predictions': df_yes, 'No_predictions': df_no}

    #now loop thru and put each on a specific sheet
    for sheet, frame in  frames.items(): # .use .items for python 3.X
        frame.to_excel(writer, sheet_name = sheet)

    #critical last step
    writer.save()
        
        
    #test1.to_excel('result.xlsx')
    
    return render_template("upload_printed.html",tables=[disp.to_html(classes='data')], titles=disp.columns.values[-1:])
#Routing to page when Attribute Entry is selected
@app.route('/attribute_entry')
def attribute_entry():
    return render_template('attribute_entry.html')

#Obtaining values from attribute entry and processing them
@app.route('/yes', methods=['GET', 'POST'])
def yes():
    #Obtaining the values from HTML form
    
    
    age=int(request.form['age'])
    dfh=int(request.form['dfh'])
    ncw=int(request.form['ncw'])
    twy=int(request.form['twy'])
    ylp=int(request.form['ylp'])
    yac=int(request.form['yac'])
    ycr=int(request.form['ycr'])
    ycm=int(request.form['ycm'])
    tly=int(request.form['tly'])
    shp=int(request.form['shp'])
    mi=int(request.form['mi'])
    ji=request.form['ji']
    jl=request.form['jl']
    ot=request.form['ot']
    bt=request.form['bt']
    jr=request.form['jr']
    el=request.form['el']
    ms=request.form['ms']
    ef=request.form['ef'] 
    sol=request.form['sol']
    pr=int(request.form['pr'])
    #print(age,'\n',dfh,'\n',ncw,'\n',twy,'\n',ylp,'\n',yac,'\n',ycr,'\n',ycm,'\n',tly,'\n',
          #shp,'\n',mi,'\n',ji,'\n',jl,'\n',ot,'\n',bt,'\n',jr,'\n',el,'\n',ms,'\n',ef,'\n',sol,'\n',pr)
    
    #Initializing the one hot encoded columns to 0
    ms_S=0
    ms_M=0
    ms_D=0

    ef_HR=0
    ef_TD=0
    ef_LS=0
    ef_Ma=0
    ef_Me=0
    ef_O=0

    jr_HCR=0
    jr_HR=0
    jr_LT=0
    jr_M=0
    jr_MD=0
    jr_RD=0
    jr_RS=0
    jr_SE=0
    jr_SR=0
    
    bt_NT=0
    bt_TF=0
    bt_TR=0

    ji_1=0
    ji_2=0
    ji_3=0
    ji_4=0

    ot_N=0
    ot_Y=0

    sol_0=0
    sol_1=0
    sol_2=0
    sol_3=0

    jl_1=0
    jl_2=0
    jl_3=0
    jl_4=0
    jl_5=0
    
    el_1=0
    el_2=0
    el_3=0
    el_4=0
    el_5=0
    
    #Setting the value obtained from form to 1
    if(ms=="1"):
        ms_S=1
    elif(ms=="2"):
        ms_M=1
    else:
        ms_D=1
        
    if(ef=="1"):
        ef_HR=1
    elif(ef=="2"):
        ef_TD=1
    elif(ef=="3"):
        ef_LS=1
    elif(ef=="4"):
        ef_Ma=1
    elif(ef=="5"):
        ef_Me=1
    else:
        ef_O=1
        
    if(jr=="1"):
        jr_HCR=1
    elif(jr=="2"):
        jr_HR=1
    elif(jr=="3"):
        jr_LT=1
    elif(jr=="4"):
        jr_M=1
    elif(jr=="5"):
        jr_MD=1
    elif(jr=="6"):
        jr_RD=1
    elif(jr=="7"):
        jr_RS=1
    elif(jr=="8"):
        jr_SE=1
    else:
        jr_SR=1
    
    if(bt=="0"):
        bt_NT=1
    elif(bt=="1"):
        bt_TR=1
    else:
        bt_TF=1
        
    if(ji=="1"):
        ji_1=1
    elif(ji=="2"):
        ji_2=1
    elif(ji=="3"):
        ji_3=1
    else:
        ji_4=1
        
    if(ot=="1"):
        ot_Y=1
    else:
        ot_N=1
        
    if(sol=="0"):
        sol_0=1
    elif(sol=="1"):
        sol_1=1
    elif(sol=="2"):
        sol_2=1
    else:
        sol_3=1
    
    if(jl=="1"):
        jl_1=1
    elif(jl=="2"):
        jl_2=1
    elif(jl=="3"):
        jl_3=1
    elif(jl=="4"):
        jl_4=1
    else:
        jl_5=1
        
    if(el=="1"):
        el_1=1
    elif(el=="2"):
        el_2=1
    elif(el=="3"):
        el_3=1
    elif(el=="4"):
        el_4=1
    else:
        el_5=1
       
    #Training the data
    train=pd.read_excel('trainfile.xlsx')
    
    train['TenurePerJob']=0
    for i in range(0,len(train)):
        if train.loc[i,'NumCompaniesWorked']>0:
            train.loc[i,'TenurePerJob']=train.loc[i,'TotalWorkingYears']/train.loc[i,'NumCompaniesWorked']
    a=np.median(train['MonthlyIncome'])
    train['CompRatioOverall']=train['MonthlyIncome']/a
            
    tpj=0
    if(ncw>0):
        tpj=twy/ncw
    cro=mi/a
    
    pmp=(pr+shp)/2
        
    #train.drop(labels=['EmployeeCount','EmployeeNumber','Over18','StandardHours'],axis=1,inplace=True)

    full_col_names=train.columns.tolist()
    num_col_names=train.select_dtypes(include=[np.int64,np.float64]).columns.tolist()
    num_cat_col_names=['Education','JobInvolvement','JobLevel','StockOptionLevel']
    target=['Attrition']
    num_col_names=list(set(num_col_names)-set(num_cat_col_names))
    cat_col_names=list(set(full_col_names)-set(num_col_names)-set(target))
    #print("total no of numerical features:",len(num_col_names))
    #print("total no of categorical & ordered features:",len(cat_col_names))
    cat_train=train[cat_col_names]
    num_train=train[num_col_names]

    for col in num_col_names:
        if num_train[col].skew()>0.80:
            num_train[col]=np.log1p(num_train[col])
        
    for col in cat_col_names:
        col_dummies=pd.get_dummies(cat_train[col],prefix=col)
        cat_train=pd.concat([cat_train,col_dummies],axis=1)
    
    Attrition={'Yes':1,'No':0}
    train.Attrition=[Attrition[item] for item in train.Attrition]

    cat_train.drop(cat_col_names,axis=1,inplace=True)

    final_train=pd.concat([num_train,cat_train],axis=1)

    final_train['pr_mean_psh'] = final_train['PerformanceRating'].add(final_train['PercentSalaryHike'])
    final_train['pr_mean_psh']=final_train['pr_mean_psh']/2
    #final_train.drop(labels=['PerformanceRating','PercentSalaryHike'],axis=1,inplace=True)
    #final_train.drop(labels=['HourlyRate','MonthlyRate','DailyRate'],axis=1,inplace=True)
    #final_train.drop(labels=['Gender_Male','Gender_Female'],axis=1,inplace=True)
    #final_train.drop(labels=['Department_Human Resources','Department_Research & Development','Department_Sales',],axis=1
    #                ,inplace=True)
    #final_train.drop(labels=['WorkLifeBalance_1','WorkLifeBalance_2','WorkLifeBalance_3','WorkLifeBalance_4'
    #                         ,'RelationshipSatisfaction_1','RelationshipSatisfaction_2','RelationshipSatisfaction_3'
    #                        ,'RelationshipSatisfaction_4','JobSatisfaction_1','JobSatisfaction_2','JobSatisfaction_3'
    #                       ,'JobSatisfaction_4','EnvironmentSatisfaction_1','EnvironmentSatisfaction_2'
    #                      ,'EnvironmentSatisfaction_3','EnvironmentSatisfaction_4'],axis=1,inplace=True)
    
    ftr=final_train
    ftr=ftr[['YearsWithCurrManager','YearsInCurrentRole','Age','YearsSinceLastPromotion','DistanceFromHome','MonthlyIncome'
             ,'NumCompaniesWorked','YearsAtCompany','TotalWorkingYears','CompRatioOverall','TrainingTimesLastYear'
             ,'TenurePerJob','EducationField_Human Resources','EducationField_Life Sciences','EducationField_Marketing'
             ,'EducationField_Medical','EducationField_Other','EducationField_Technical Degree','BusinessTravel_Non-Travel'
             ,'BusinessTravel_Travel_Frequently','BusinessTravel_Travel_Rarely','JobRole_Healthcare Representative'
             ,'JobRole_Human Resources','JobRole_Laboratory Technician','JobRole_Manager','JobRole_Manufacturing Director'
             ,'JobRole_Research Director','JobRole_Research Scientist','JobRole_Sales Executive','JobRole_Sales Representative'
             ,'MaritalStatus_Divorced','MaritalStatus_Married','MaritalStatus_Single','StockOptionLevel_0','StockOptionLevel_1'
             ,'StockOptionLevel_2','StockOptionLevel_3','Education_1','Education_2','Education_3','Education_4','Education_5'
             ,'JobLevel_1','JobLevel_2','JobLevel_3','JobLevel_4','JobLevel_5','JobInvolvement_1','JobInvolvement_2'
             ,'JobInvolvement_3','JobInvolvement_4','OverTime_No','OverTime_Yes','pr_mean_psh']]
    
    ev_list=[ycm,ycr,age,ylp,dfh,mi,ncw,yac,twy,cro,tly,tpj,ef_HR,ef_LS,ef_Ma,ef_Me,ef_O,ef_TD,bt_NT,bt_TF,bt_TR
        ,jr_HCR,jr_HR,jr_LT,jr_M,jr_MD,jr_RD,jr_RS,jr_SE,jr_SR,ms_D,ms_M,ms_S,sol_0,sol_1,sol_2,sol_3
        ,el_1,el_2,el_3,el_4,el_5,jl_1,jl_2,jl_3,jl_4,jl_5,ji_1,ji_2,ji_3,ji_4,ot_N,ot_Y,pmp]
    
    #ev_list=ev.tolist()
    evdf=pd.DataFrame([ev_list],columns=list(ftr))
    norm=ftr.append(evdf,ignore_index=True)
    for i in list(norm):
        norm[i]=norm[i]/norm[i].max()
    final_norm_ev=norm.tail(1)
    #final_norm_ev.to_excel('final_norm_ev.xlsx')
    #final_norm_train=norm.drop(norm.tail(1).index,inplace=True)
    norm.drop(norm.tail(1).index,inplace=True)
    #norm.to_excel('norm.xlsx')
    #print(norm.head())
    X_train=np.asarray(norm)
    Y_train=np.asarray(train['Attrition'])
    
    
    
    X_test=np.asarray(final_norm_ev)
    #print(X_train.shape)
    #print(X_test.shape)
    
    lr=LogisticRegression(solver='liblinear').fit(X_train,Y_train)
    yhat=lr.predict(X_test)
    rp=lr.predict_proba(X_test).max()
    rp=rp*100
    rp=round(rp,2)
    #print(rp)
    #print(yhat)
    if(yhat==1):
        disp='Yes'
    else:
        disp='No' 
        
    #added affecting parameters here
    
    #l=np.abs(lr.coef_).tolist()
    #print(l)
    coefs = [0.873696131333615,
 0.8931174132179643,
 1.0994873699061312,
 1.0959319112190686,
 1.3012173197470709,
 0.41642695006597896,
 1.3174481693002256,
 0.44078711574482826,
 1.2116346585265028,
 0.6838631013266984,
 0.877363617559784,
 0.4359599425225076,
 0.24018626782579783,
 0.2494013777908805,
 0.22983112845974615,
 0.35469996834108175,
 0.4390193042070946,
 0.5015026383536926,
 0.7992061117861377,
 0.8311287980994192,
 0.10352330201314955,
 0.012105992066855897,
 0.06672391364467412,
 0.4066083125238653,
 0.07810293138802382,
 0.05874767962635775,
 1.1219206800765629,
 0.5449851546723561,
 0.9434003236564056,
 0.30331728817169973,
 0.1488516483658626,
 0.005938614411119775,
 0.08318964707714606,
 0.6745102679497458,
 0.5248570110434769,
 0.4020707554155931,
 0.180816882809567,
 0.24335291976765855,
 0.06427698831019184,
 0.001919413661863494,
 0.0768797132053018,
 0.02867618889044523,
 0.6143432814525115,
 0.8415046804355878,
 0.19466280682742523,
 0.5353264860427034,
 0.496224462498549,
 0.9970872013487172,
 0.028198420970424333,
 0.30527751810035925,
 0.7352118779776915,
 0.8500754513412074,
 0.7784748356413683,
 0.08503746009862677]



    #print(coefs)
    #print(np.asarray(final_norm_ev).tolist())
    data=np.asarray(final_norm_ev).tolist()
    
    #print(data)
    
    weights=np.multiply(data,coefs).tolist()
    #print(weights)
    
    cols=list(ftr)
    L=[]
    for val in weights:
        dic = dict(enumerate(val))
        L.append(dic)
    #print(L)

    ColWeights=[]
    for dic in L:
        i=0
        tempDic={}
        for key,value in dic.items():
            key=cols[i]
            tempDic[key]=value
            i=i+1
        ColWeights.append(tempDic)
    #print(ColWeights)
     
    changable_cols=['YearsWithCurrManager',
                    'MonthlyIncome',
                    'YearsInCurrentRole',
                    'DistanceFromHome',
                    'YearsSinceLastPromotion',
                    'JobLevel_1',
                    'JobLevel_2',
                    'JobLevel_3',
                    'JobLevel_4',
                    'BusinessTravel_Non-Travel',
                    'BusinessTravel_Travel_Frequently',
                    'BusinessTravel_Travel_Rarely',
                    'OverTime_Yes']

    Col_Weights=[]
    for dic in ColWeights:
        a={}
        for k,v in dic.items():
            if k in changable_cols :
                a[k]=v
        Col_Weights.append(a) 
    #print(Col_Weights)
       
    AscendingCols=[]
    for dic in Col_Weights:
        AscendingCols.append((sorted(dic, key=dic.get)))
    #print(AscendingCols)
        
    a1='YearsWithCurrManager'
    a2='MonthlyIncome'
    a3='YearsInCurrentRole'
    a4='DistanceFromHome'
    a5='YearsSinceLastPromotion'
    a6='JobLevel_1'
    a7='JobLevel_2'
    a8='JobLevel_3'
    a9='JobLevel_4'
    a10='BusinessTravel_Non-Travel'
    a11='BusinessTravel_Travel_Frequently'
    a12='BusinessTravel_Travel_Rarely'
    a13='OverTime_Yes'
    #a14='OverTime_No'
        
    AllParams=[]
    for h in AscendingCols:
        params=[ h[12], h[11], h[10], h[9], h[8] ]
        AllParams.append(params)
   
    AllParams=[item for sublist in AllParams for item in sublist]
    #print(AllParams)
    #print(AllParam)
 
    
    b1=0
    b2=0
    b3=0
    b4=0
    b5=0
    b6=0
    b7=0
    b8=0
    b9=0
    b10=0
    b11=0
    b12=0
    b13=0
    #b14=0
    
    if a1 in AllParams:
        b1=1
    if a2 in AllParams:
        b2=1
    if a3 in AllParams:
        b3=1
    if a4 in AllParams:
        b4=1
    if a5 in AllParams:
        b5=1
    if a6 in AllParams:
        b6=1
    if a7 in AllParams:
        b7=1
    if a8 in AllParams:
        b8=1
    if a9 in AllParams:
        b9=1
    if a10 in AllParams:
        b10=1
    if a11 in AllParams:
        b11=1
    if a12 in AllParams:
        b12=1
    if a13 in AllParams:
        b13=1
    #if a14 in AllParams:
        #b14=1
    if disp=='Yes':
        #print(b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13)        
        return render_template('yes.html',value=disp,value1=rp,b1=b1,b2=b2,b3=b3,b4=b4,b5=b5,b6=b6,b7=b7
                           ,b8=b8,b9=b9,b10=b10,b11=b11,b12=b12,b13=b13,ycm=ycm,ycr=ycr,age=age,ylp=ylp,dfh=dfh,mi=mi
                           ,ncw=ncw,yac=yac,twy=twy,cro=cro,tly=tly,tpj=tpj,ef_HR=ef_HR,ef_LS=ef_LS,ef_Ma=ef_Ma,ef_Me=ef_Me
                           ,ef_O=ef_O,ef_TD=ef_TD,bt_NT=bt_NT,bt_TF=bt_TF,bt_TR=bt_TR,jr_HCR=jr_HCR,jr_HR=jr_HR,jr_LT=jr_LT
                           ,jr_M=jr_M,jr_MD=jr_MD,jr_RD=jr_RD,jr_RS=jr_RS,jr_SE=jr_SE,jr_SR=jr_SR,ms_D=ms_D,ms_M=ms_M,ms_S=ms_S
                           ,sol_0=sol_0,sol_1=sol_1,sol_2=sol_2,sol_3=sol_3,el_1=el_1,el_2=el_2,el_3=el_3,el_4=el_4,el_5=el_5
                           ,jl_1=jl_1,jl_2=jl_2,jl_3=jl_3,jl_4=jl_4,jl_5=jl_5,ji_1=ji_1,ji_2=ji_2,ji_3=ji_3,ji_4=ji_4
                           ,ot_N=ot_N,ot_Y=ot_Y,pmp=pmp,jl=jl,shp=shp,pr=pr,ji=ji,jr=jr,el=el,ms=ms,ef=ef,sol=sol)
    else:
        aff_params_no=['YearsWithCurrManager',
                           'YearsInCurrentRole',
                           'MonthlyIncome',
                           'YearsAtCompany',
                           'TotalWorkingYears',]
    
            #MAIN PARAMS FOR NO
        Col_Weights_No=[]
        for dic in ColWeights:
                b={}
                for k,v in dic.items():
                    if k in aff_params_no :
                        b[k]=v
                Col_Weights_No.append(b)    
            
        
        AscendingCols1=[]
        for dic in Col_Weights_No:
                AscendingCols1.append((sorted(dic, key=dic.get)))
        
        for h in AscendingCols1:
                a=h[4]
                b=h[3]
                c=h[2]
                d=h[1]
                e=h[0]
        
        return render_template('no.html',value=disp,value1=rp,a=a,b=b,c=c,d=d,e=e)


if __name__ == '__main__':
    app.run()
