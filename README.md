# 🎓 Student Dropout Early Warning System

## 📋 Project Overview

A machine learning-based early warning system that identifies students at risk of dropping out, enabling proactive intervention by academic advisors.

### 🎯 Problem Statement
- Universities lose students each semester due to various risk factors
- Need for early identification to enable timely intervention
- Advisors require interpretable, actionable insights

### 💡 Solution
- ML model predicting dropout risk using early-semester data
- Risk scoring system (Low/Medium/High)
- Interactive dashboard for advisors
- Actionable recommendations per risk level

## 🔧 Technical Implementation

### 1. Data Cleaning
```python
- Removed missing values (none found in dataset)
- Standardized column names
- Encoded target variable (Dropout=1, Continue=0)
- Created synthetic student IDs for tracking'''

### 2. Features Used (Early Semester Only)
'''Numerical Features:

raised_hands: Class participation (0-100)

visited_resources: Learning material access (0-100)

announcements_view: Announcement engagement (0-100)

discussion_posts: Discussion forum activity (0-100)

Engineered Features:

engagement_score: Weighted composite of participation metrics

low_engagement: Binary flag for engagement < 30%

parent_involvement: Score based on survey responses

early_semester_proxy: Early success indicator

Categorical Features:

Demographics (gender, nationality, birth place)

Academic (stage, grade, section, topic, semester)

Parent-related (relationship, survey responses, satisfaction)

Attendance patterns'''

###3. Model Selection
'''Algorithm: Random Forest Classifier

Why Random Forest?

Handles mixed data types well

Provides feature importance for interpretability

Resistant to overfitting

Captures non-linear relationships'''
###4. Performance Metrics
''' Metric	                    Score
    ROC-AUC                     0.85+
    Precision (High Risk)	    ~75%
    Recall (High Risk)	        ~70%
    False Positive Rate     	~15%'''
###5. Risk Thresholds
'''Risk levels calibrated to balance early detection vs. false alarms:

Risk Level          Probability         Action          Intervention
Low                 0-30%	            Monitor     	Monthly check-in
Medium	            30-60%	            Watch       	Bi-weekly support
High	            60-100%	            Intervene   	Immediate counseling'''
