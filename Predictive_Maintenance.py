
import pandas as pd

telemetry = pd.read_csv('data/PdM_telemetry.csv')
errors = pd.read_csv('data/PdM_errors.csv')
maint = pd.read_csv('data/PdM_maint.csv')
failures = pd.read_csv('data/PdM_failures.csv')
machines = pd.read_csv('data/PdM_machines.csv')

# Preprocessing : Findout the histogram of machines ni 0-5,5-10,10-15,15-20
import matplotlib.pyplot as plt
x=machines['age']
plt.hist(x, bins=10)
plt.show()

#plot time vs machineid for failures
plt.rcParams["figure.figsize"] = (100,50)
plt.scatter(failures['datetime'],failures['machineID'])
plt.show()
#plot time vs machineid for maintenance
plt.rcParams["figure.figsize"] = (100,50)
plt.scatter(maint['datetime'],maint['machineID'])
plt.show()

# format datetime field which comes in as string
telemetry['datetime'] = pd.to_datetime(telemetry['datetime'], format="%Y-%m-%d %H:%M:%S")
print("Total number of telemetry records: %d" % len(telemetry.index))
print(telemetry.head())
telemetry.describe()

# format of datetime field which comes in as string
errors['datetime'] = pd.to_datetime(errors['datetime'],format = '%Y-%m-%d %H:%M:%S')
errors['errorID'] = errors['errorID'].astype('category')
print("Total Number of error records: %d" %len(errors.index))
errors.head()
errors['errorID'].value_counts()


maint['datetime'] = pd.to_datetime(maint['datetime'], format='%Y-%m-%d %H:%M:%S')
maint['comp'] = maint['comp'].astype('category')
print("Total Number of maintenance Records: %d" %len(maint.index))
maint.head()
maint['comp'].value_counts()

machines['model'] = machines['model'].astype('category')
print("Total number of machines: %d" % len(machines.index))
machines.head()


# format datetime field which comes in as string
failures['datetime'] = pd.to_datetime(failures['datetime'], format="%Y-%m-%d %H:%M:%S")
failures['failure'] = failures['failure'].astype('category')
print("Total number of failures: %d" % len(failures.index))
failures.head()
failures['failure'].value_counts()

# Calculate mean values for telemetry features
temp = []
fields = ['volt', 'rotate', 'pressure', 'vibration']
for col in fields:
    temp.append(pd.pivot_table(telemetry,
                               index='datetime',
                               columns='machineID',
                               values=col).resample('3H', closed='left', label='right', how='mean').unstack())
telemetry_mean_3h = pd.concat(temp, axis=1)
telemetry_mean_3h.columns = [i + 'mean_3h' for i in fields]
telemetry_mean_3h.reset_index(inplace=True)

# repeat for standard deviation
temp1 = []
for col in fields:
    temp1.append(pd.pivot_table(telemetry,
                               index='datetime',
                               columns='machineID',
                               values=col).resample('3H', closed='left', label='right', how='std').unstack())
telemetry_sd_3h = pd.concat(temp1, axis=1)
telemetry_sd_3h.columns = [i + 'sd_3h' for i in fields]
telemetry_sd_3h.reset_index(inplace=True)

temp2 = []
fields = ['volt', 'rotate', 'pressure', 'vibration']
for col in fields:
    temp2.append(pd.pivot_table(telemetry,
                                               index='datetime',
                                               columns='machineID',
                                               values=col).rolling(24).mean().resample('3H',
                                                                                closed='left',
                                                                                label='right',
                                                                                how='first').unstack())
    
telemetry_mean_24h = pd.concat(temp2, axis=1)
telemetry_mean_24h.columns = [i + 'mean_24h' for i in fields]
telemetry_mean_24h.reset_index(inplace=True)
telemetry_mean_24h = telemetry_mean_24h.loc[-telemetry_mean_24h['voltmean_24h'].isnull()]

# repeat for standard deviation
temp3 = []
fields = ['volt', 'rotate', 'pressure', 'vibration']
for col in fields:
    temp3.append(pd.pivot_table(telemetry,
                                               index='datetime',
                                               columns='machineID',
                                               values=col).rolling(24).std().resample('3H',
                                                                                closed='left',
                                                                                label='right',
                                                                                how='first').unstack())
telemetry_sd_24h = pd.concat(temp3, axis=1)
telemetry_sd_24h.columns = [i + 'sd_24h' for i in fields]
telemetry_sd_24h = telemetry_sd_24h.loc[-telemetry_sd_24h['voltsd_24h'].isnull()]
telemetry_sd_24h.reset_index(inplace=True)

telemetry_mean_24h.head(10)

# merge columns of feature sets created earlier
telemetry_feat = pd.concat([telemetry_mean_3h,
                            telemetry_sd_3h.ix[:, 2:6],
                            telemetry_mean_24h.ix[:, 2:6],
                            telemetry_sd_24h.ix[:, 2:6]], axis=1).dropna()
telemetry_feat.describe()

telemetry_feat.head()

# create a column for each error type
error_count = pd.get_dummies(errors.set_index('datetime')).reset_index()
error_count
error_count.columns = ['datetime', 'machineID', 'error1', 'error2', 'error3', 'error4', 'error5']
error_count.head(13)

# combine errors for a given machine in a given hour
error_count = error_count.groupby(['machineID','datetime']).sum().reset_index()
error_count.head(13)

error_count = telemetry[['datetime', 'machineID']].merge(error_count, on=['machineID', 'datetime'], how='left').fillna(0.0)
error_count.describe()

temp4 = []
fields = ['error%d' % i for i in range(1,6)]
for col in fields:
    temp4.append(pd.pivot_table(error_count,
                                               index='datetime',
                                               columns='machineID',
                                               values=col).rolling(24).sum().resample('3H',
                                                                             closed='left',
                                                                             label='right',
                                                                             how='first').unstack())
    
error_count = pd.concat(temp4, axis=1)
error_count.columns = [i + 'count' for i in fields]
error_count.reset_index(inplace=True)
error_count = error_count.dropna()
error_count.describe()

error_count.head()

import numpy as np

# create a column for each error type
comp_rep = pd.get_dummies(maint.set_index('datetime')).reset_index()
comp_rep.columns = ['datetime', 'machineID', 'comp1', 'comp2', 'comp3', 'comp4']

# combine repairs for a given machine in a given hour
comp_rep = comp_rep.groupby(['machineID', 'datetime']).sum().reset_index()

# add timepoints where no components were replaced
comp_rep = telemetry[['datetime', 'machineID']].merge(comp_rep,
                                                      on=['datetime', 'machineID'],
                                                      how='outer').fillna(0).sort_values(by=['machineID', 'datetime'])

components = ['comp1', 'comp2', 'comp3', 'comp4']
for comp in components:
    # convert indicator to most recent date of component change
    comp_rep.loc[comp_rep[comp] < 1, comp] = None
    comp_rep.loc[-comp_rep[comp].isnull(), comp] = comp_rep.loc[-comp_rep[comp].isnull(), 'datetime']
    
    # forward-fill the most-recent date of component change
    comp_rep[comp] = comp_rep[comp].fillna(method='ffill')

# remove dates in 2014 (may have NaN or future component change dates)    
comp_rep = comp_rep.loc[comp_rep['datetime'] > pd.to_datetime('2015-01-01')]

# replace dates of most recent component change with days since most recent component change
for comp in components:
    comp_rep[comp] = (pd.to_timedelta(comp_rep['datetime']) - (pd.to_timedelta(comp_rep[comp]))) /np.timedelta64(1, 'D')
    

telemetry_feat

final_feat = telemetry_feat.merge(error_count, on=['datetime', 'machineID'], how='left')
final_feat = final_feat.merge(comp_rep, on=['datetime', 'machineID'], how='left')
final_feat = final_feat.merge(machines, on=['machineID'], how='left')

print(final_feat.head())
final_feat.describe()

labeled_features = final_feat.merge(failures, on=['datetime', 'machineID'], how='left')
labeled_features = labeled_features.fillna(method='bfill', limit=7,axis=1) # fill backward up to 24h
labeled_features = labeled_features.fillna('none')

X=pd.get_dummies(labeled_features.drop(['failure','datetime','machineID'],axis=1))
Y=labeled_features['failure']

feat_labels=list(X.columns)

######### RF feature selection ####################################3
# Split the data 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Create a random forest classifier
clf = RandomForestClassifier(n_estimators=100,random_state=0)

# Train the classifier
clf.fit(X_train, y_train)


# Print the name and gini importance of each feature
feature=[]
for i in zip(feat_labels, clf.feature_importances_):
    feature.append(i)
    
# Create a selector object that will use the random forest classifier to identify
# features that have an importance of more than 0.01
scores = clf.feature_importances_
sfm = SelectFromModel(clf, threshold=0.01)

# Train the selector
sfm.fit(X_train, y_train)
# Print the names of the most important features
main_features=[]
for j in sfm.get_support(indices=True):
    main_features.append(feat_labels[j])

top_feat=list(main_features)
top_feat1=labeled_features[top_feat]



'''from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(top_13)
rescaledX = scaler.transform(top_13)'''

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(top_feat1, Y, test_size=0.2, random_state=30)

######################## Decision Tree Classiier ###########################
from sklearn.tree import DecisionTreeClassifier
dec_classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
dec_classifier.fit(X_train, y_train)

#############################################################################

######################## Random Forest Classifier #########################
from sklearn.ensemble import RandomForestClassifier
rand_classifier=RandomForestClassifier(n_estimators=100,random_state=15,n_jobs=-1)
rand_classifier.fit(X_train, y_train)
###########################################################################

######################## Xg Boost Classifier ###############################
from xgboost import XGBClassifier
xg_classifier = XGBClassifier(n_estimators=100, max_depth=6, silent=False)
xg_classifier.fit(X_train, y_train)

###########################################################################

##################### Logistic Regression ##################################
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

############################################################################

########## Predicting the Train set results #################################
y_pred_dec_train = dec_classifier.predict(X_train)
y_pred_rand_train=rand_classifier.predict(X_train)
y_pred_xg_train=xg_classifier.predict(X_train)
y_pred_logreg_train=logreg.predict(X_train)

############################################################################

########## Predicting the Test set results #################################
y_pred_dec = dec_classifier.predict(X_test)
y_pred_rand=rand_classifier.predict(X_test)
y_pred_xg=xg_classifier.predict(X_test)
y_pred_logreg=logreg.predict(X_test)

############################################################################

################ Making the Confusion Matrix ###############################
from sklearn.metrics import confusion_matrix
dec_cm = confusion_matrix(y_test, y_pred_dec)
rand_cm = confusion_matrix(y_test, y_pred_rand)
xg_cm = confusion_matrix(y_test, y_pred_xg)
logreg_cm = confusion_matrix(y_test, y_pred_logreg)

############################################################################

############## Accuracy #################################

dec_accuracy=np.array([float(np.trace(dec_cm)) / np.sum(dec_cm)]*5)
rand_accuracy=np.array([float(np.trace(rand_cm)) / np.sum(rand_cm)]*5)
xg_accuracy=np.array([float(np.trace(xg_cm)) / np.sum(xg_cm)]*5)
logreg_accuracy=np.array([float(np.trace(logreg_cm)) / np.sum(logreg_cm)]*5)


############################################################################

############### Precision  tp / (tp + fp)###################################
from sklearn.metrics import precision_score
dec_precision = precision_score(y_test, y_pred_dec,average=None)
rand_precision = precision_score(y_test, y_pred_rand,average=None)
xg_precision = precision_score(y_test, y_pred_xg,average=None)
logreg_precision = precision_score(y_test, y_pred_logreg,average=None)


############################################################################

################ Recall: tp / (tp + fn) ###################################
from sklearn.metrics import recall_score
dec_recall = recall_score(y_test,y_pred_dec,average=None)
rand_recall = recall_score(y_test,y_pred_rand,average=None)
xg_recall = recall_score(y_test,y_pred_xg,average=None)
logreg_recall = recall_score(y_test,y_pred_logreg,average=None)


############################################################################

################# F1: 2 *(precision*recall)/ (precision+recall) ############
from sklearn.metrics import f1_score
dec_f1 = f1_score(y_test,y_pred_dec,average=None)
rand_f1 = f1_score(y_test,y_pred_rand,average=None)
xg_f1 = f1_score(y_test,y_pred_xg,average=None)
logreg_f1 = f1_score(y_test,y_pred_logreg,average=None)


#############################################################################
rand_results = {'None':[rand_accuracy[0], rand_precision[0], rand_recall[0], rand_f1[0]] ,
        'Comp1':[rand_accuracy[1], rand_precision[1], rand_recall[1], rand_f1[1]],
        'Comp2':[rand_accuracy[2], rand_precision[2], rand_recall[2], rand_f1[2]],
        'Comp3':[rand_accuracy[3], rand_precision[3], rand_recall[3], rand_f1[3]],
        'Comp4':[rand_accuracy[4], rand_precision[4], rand_recall[4], rand_f1[4]]
        }
rand_df = pd.DataFrame(rand_results, columns = ['None', 'Comp1','Comp2','Comp3','Comp4'])
rand_df.index = ['Accuracy', 'Precision', 'Recall', 'F1_Score'] 

#############################################################################
dec_results = {'None':[dec_accuracy[0], dec_precision[0], dec_recall[0], dec_f1[0]] ,
        'Comp1':[dec_accuracy[1], dec_precision[1], dec_recall[1], dec_f1[1]],
        'Comp2':[dec_accuracy[2], dec_precision[2], dec_recall[2], dec_f1[2]],
        'Comp3':[dec_accuracy[3], dec_precision[3], dec_recall[3], dec_f1[3]],
        'Comp4':[dec_accuracy[4], dec_precision[4], dec_recall[4], dec_f1[4]]
        }

dec_df = pd.DataFrame(dec_results, columns = ['None', 'Comp1','Comp2','Comp3','Comp4'])
dec_df.index = ['Accuracy', 'Precision', 'Recall', 'F1_Score'] 


xg_results = {'None':[xg_accuracy[0], xg_precision[0], xg_recall[0], xg_f1[0]] ,
        'Comp1':[xg_accuracy[1], xg_precision[1], xg_recall[1], xg_f1[1]],
        'Comp2':[xg_accuracy[2], xg_precision[2], xg_recall[2], xg_f1[2]],
        'Comp3':[xg_accuracy[3], xg_precision[3], xg_recall[3], xg_f1[3]],
        'Comp4':[xg_accuracy[4], xg_precision[4], xg_recall[4], xg_f1[4]]
        }
xg_df = pd.DataFrame(xg_results, columns = ['None', 'Comp1','Comp2','Comp3','Comp4'])
xg_df.index = ['Accuracy', 'Precision', 'Recall', 'F1_Score'] 


logreg_results = {'None':[logreg_accuracy[0], logreg_precision[0], logreg_recall[0], logreg_f1[0]] ,
        'Comp1':[logreg_accuracy[1], logreg_precision[1], logreg_recall[1], logreg_f1[1]],
        'Comp2':[logreg_accuracy[2], logreg_precision[2], logreg_recall[2], logreg_f1[2]],
        'Comp3':[logreg_accuracy[3], logreg_precision[3], logreg_recall[3], logreg_f1[3]],
        'Comp4':[logreg_accuracy[4], logreg_precision[4], logreg_recall[4], logreg_f1[4]]
        }
logreg_df = pd.DataFrame(logreg_results, columns = ['None', 'Comp1','Comp2','Comp3','Comp4'])
logreg_df.index = ['Accuracy', 'Precision', 'Recall', 'F1_Score'] 
