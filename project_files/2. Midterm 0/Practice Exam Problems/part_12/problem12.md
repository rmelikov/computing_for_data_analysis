# Problem 12: Job Recommendation

In this problem, you will recommend jobs to a person based on his previous job application history, which is called [content-based filtering](https://en.wikipedia.org/wiki/Recommender_system#Content-based_filtering). The problem will test your knowledge of basic Python data structures, such as lists, dictionaries (with nested dictionaries), default dictionaries, and sorting.

There are 5 exercises in this problem, numbered 0 through 4 and worth 10 points in total.

## Loading and interpreting the data

Please run the following code cell to load the data. It will create two data structures:

1. **`People`**: This object is a nested dictionary that contains people and their job histories.
2. **`Hiring_jobs`**: This object is also dictionary. It contains open positions.

The code cell will also print some of this data, after which we'll explain the format in more detail.


```python
# Load the data
import json

with open('people.json') as people_data:
    People = json.load(people_data,)

with open('hiringjobs.json') as jobs_data:
    Hiring_jobs = json.load(jobs_data,)
    
print("* Use People['0'] to access the first person's job history ==>\nPeople[0] == {}".format(People['0']))
print("\n* Use People['1'] to access the second person's job history ==>\nPeople[1] == {}".format(People['1']))
print('\n(... and so on ...)\n')

print("Use People['0']['1'] to access the first person's second job history ==>\nPeople['0']['1'] == {}".format(People['0']['1']))
print('\n')

print("Hiring_jobs ==>\n{}".format(Hiring_jobs))
```

    * Use People['0'] to access the first person's job history ==>
    People[0] == {'0': {'title': 'data scientist', 'company': 'apple', 'status': 'apply'}, '1': {'title': 'data analyst', 'company': 'google', 'status': 'apply'}, '2': {'title': 'postdoc', 'company': 'georgia tech', 'status': 'current'}, '3': {'title': 'research assistant', 'company': 'georgia tech', 'status': 'previous'}}
    
    * Use People['1'] to access the second person's job history ==>
    People[1] == {'0': {'title': 'data scientist', 'company': 'google', 'status': 'apply'}, '1': {'title': 'data engineer', 'company': 'amazon', 'status': 'apply'}, '2': {'title': 'data scientist', 'company': 'facebook', 'status': 'apply'}, '3': {'title': 'data scientist', 'company': 'microsoft', 'status': 'apply'}, '4': {'title': 'data analyst', 'company': 'chase', 'status': 'previous'}}
    
    (... and so on ...)
    
    Use People['0']['1'] to access the first person's second job history ==>
    People['0']['1'] == {'title': 'data analyst', 'company': 'google', 'status': 'apply'}
    
    
    Hiring_jobs ==>
    {'0': ['data scientist', 'IBM'], '1': ['data scientist', 'coca cola'], '2': ['data analyst', 'cnn'], '3': ['data scientist', 'google'], '4': ['business analyst', 'chase'], '5': ['data scientist', 'apple'], '6': ['scientist', 'biogen'], '7': ['scientist', 'broad institute'], '8': ['postdoc', 'mit'], '9': ['data engineer', 'microsoft']}
    

In the `People` dictionary, each key is a person's ID and the person's job history is the corresponding value. This history is, itself, represented as a dictionary, where each key is a history ID and the associated value is another dictionary whose keys are `'title'`, `'company'`, and `'status'`. The latter field, `'status'`, indicates the person's status with respect to the job. It can be `'apply'`, meaning the person has applied for the job and is waiting to hear back; `'current'`, meaning the person is currently working that job; or `'previous'`, meaning the person had previously worked at that job but no longer does so.

In the `Hiring_jobs` dictionary, each key is a job ID. Each corresponding value is an ordered list with containing two values, a job title and a company name.

For instance, consider the above output. Person 0 (ID of `'0'`) has a job history with four entries. In the case of job entries `People['0']['0']` (data scientist position at Apple) and `People['0']['1']` (data analyst position at Google), the person's status is `'apply'`, meaning that he or she has applied for the job and is awaiting a decision. In `People['0']['2']`, the status is `'current'` meaning it is the person's current position (postdoc at Georgia Tech).

## Exercises

**Exercise 0** (1 point). Given the `People` dictionary and a person's ID, `pid`, find all jobs that the person has applied for. Return these as a list. Each entry of this list represents a job as a pair (2-tuple) whose fields are the job's title and company. If the person has no active applications, the function should return an empty list. 

For example, for person with ID '0', he/she has applied for two jobs: `'data scientist'` in `'apple'` and `'data analyst'` in `'google'`. In this case, your function should return a list containing two tuples:

```python
    find_applications(People, '0') == [('data scientist', 'apple'), ('data analyst', 'google')]
```


```python
def find_applications(People, pid):

    assert type(People) is dict and len(People) != 0
    assert type(pid) is str

    def find_applications__0(People, pid):
        applied_jobs = []
        for job in People[pid].values():
            if job['status'] == 'apply':
                applied_jobs.append((job['title'], job['company']))
        return applied_jobs

    def find_applications__1(People, pid):
        #is_app = lambda job: job['status'] == 'apply'
        #get_value = lambda job: (job['title'], job['company'])
        #history = People[pid].values()
        #return [get_value(job) for job in history if is_app(job)]
        return [
            (job['title'], job['company'])
            for job in People[pid].values() 
            if job['status'] == 'apply'
        ]
        
    return find_applications__1(People, pid)
```


```python
# `find_applied_jobs_test` (1 point): Test cell
def print_applications(applied_jobs, pid):
    # print the applied job title and company for each person
    print("\nThe person with pid '{}' applied for {} jobs:".format(pid, len(applied_jobs)))
    if len(applied_jobs) >= 1:
        for i in applied_jobs:
            print("'{}' in '{}';".format(i[0], i[1]))

pid = '0'
applied_jobs = find_applications(People, pid)
assert type(applied_jobs) is list, "applied_jobs should be a list."
assert type(applied_jobs[0]) is tuple, "Each item in applied_jobs list should be a tuple."
assert len(applied_jobs) == 2, "The person with pid '0' only applied for 2 jobs."
assert ('data scientist', 'apple') in applied_jobs
assert ('data analyst', 'google') in applied_jobs
print_applications(applied_jobs, pid)

pid = '1'
applied_jobs = find_applications(People, pid)
assert type(applied_jobs) is list and type(applied_jobs[0]) is tuple
assert len(applied_jobs) == 4
assert ('data scientist', 'google') in applied_jobs and ('data engineer', 'amazon') in applied_jobs
assert ('data scientist', 'facebook') in applied_jobs 
assert ('data scientist', 'microsoft') in applied_jobs
print_applications(applied_jobs, pid)

pid = '2'
applied_jobs = find_applications(People, pid)
assert type(applied_jobs) is list
assert len(applied_jobs) == 0
print_applications(applied_jobs, pid)

pid = '3'
applied_jobs = find_applications(People, pid)
assert type(applied_jobs) is list and type(applied_jobs[0]) is tuple
assert len(applied_jobs) == 5
assert ('scientist', 'sentien biotechnologies') in applied_jobs
assert ('scientist', 'black diamond networks') in applied_jobs
assert ('scientist', 'surface oncology') in applied_jobs
assert ('scientist', 'akebia therapeutics') in applied_jobs
assert ('scientist', 'lifemine therapeutics') in applied_jobs
print_applications(applied_jobs, pid)

pid = '4'
applied_jobs = find_applications(People, pid)
assert type(applied_jobs) is list and type(applied_jobs[0]) is tuple
assert len(applied_jobs) == 5
assert ('scientist', 'dana-Farber cancer institute') in applied_jobs
assert ('scientist', 'agenus') in applied_jobs
assert ('bioinformatician', 'mitra biotech') in applied_jobs
assert ('data scientist', 'boston VA research institute') in applied_jobs
assert ('scientist', 'daley and associates') in applied_jobs
print_applications(applied_jobs, pid)

print("\n(Passed!)")
```

    
    The person with pid '0' applied for 2 jobs:
    'data scientist' in 'apple';
    'data analyst' in 'google';
    
    The person with pid '1' applied for 4 jobs:
    'data scientist' in 'google';
    'data engineer' in 'amazon';
    'data scientist' in 'facebook';
    'data scientist' in 'microsoft';
    
    The person with pid '2' applied for 0 jobs:
    
    The person with pid '3' applied for 5 jobs:
    'scientist' in 'sentien biotechnologies';
    'scientist' in 'black diamond networks';
    'scientist' in 'surface oncology';
    'scientist' in 'akebia therapeutics';
    'scientist' in 'lifemine therapeutics';
    
    The person with pid '4' applied for 5 jobs:
    'scientist' in 'dana-Farber cancer institute';
    'scientist' in 'agenus';
    'bioinformatician' in 'mitra biotech';
    'data scientist' in 'boston VA research institute';
    'scientist' in 'daley and associates';
    
    (Passed!)
    

**Exercise 1** (2 points). Implement `find_job_freq(applications)` so that it does the following. Given a list of `(title, company)` pairs, it should count how many times each `title` occurs. Then, it should return this data as a **dictionary**, where each key is a job title and the corresponding value is the count. (You may return either a standard Python `dict` or a `collections.defaultdict`.)

For example:

```python
    apps = [('data scientist', 'apple'), ('data analyst', 'google'), ('data scientist', 'microsoft')]
    find_job_freq(apps) == {'data scientist': 2, 'data analyst': 1}
```

**Note: If the person hasn't applied for any job, return an empty dictionary.** There are two test cells for this exercise, each has one point, to give you some partial credits. The test_1 cell is for general cases. The test_2 cell is to check how you deal with the not-applying-job person. 


```python
def find_job_freq(applications):
    type(applications) is list

    def my_solution():
        from collections import Counter
        return dict(
            Counter(
                [t[0] for t in applications]
            )
        )

    def find_job_freq__0():
        from collections import defaultdict
        job_freq = defaultdict(int)
        for job in applications:
            job_freq[job[0]] += 1
        return job_freq
    
    def find_job_freq__1():
        from collections import Counter
        job_titles = []
        for title, _ in applied_jobs:
            job_titles.append(title)
        job_freq = dict(Counter(job_titles))
        return job_freq
    
    return my_solution()
```


```python
# `find_job_freq_test_1` (1 point): Test cell for general cases
from collections import defaultdict

def print_job_freq(job_freq, pid):
    # print the applied job title and frequency for each person
    print("\nThe person with pid '{}' applied for {} different types of jobs:".format(pid, len(job_freq)))
    if len(job_freq) >= 1:
        for k, v in job_freq.items():
            print("{} of '{}';".format(v, k))
            
pid = '0'
applied_jobs = find_applications(People, pid)
job_freq = find_job_freq(applied_jobs)
assert type(job_freq) is dict or type(job_freq) is defaultdict, "job_freq should be a dict or defaultdict."
assert len(job_freq) == 2, "The person with pid '0' applied for 2 different types of jobs."
assert 'data scientist' in job_freq and job_freq['data scientist'] == 1
print_job_freq(job_freq, pid)

pid = '1'
applied_jobs = find_applications(People, pid)
job_freq = find_job_freq(applied_jobs)
assert type(job_freq) is dict or type(job_freq) is defaultdict
assert len(job_freq) == 2
assert 'data scientist' in job_freq and job_freq['data scientist'] == 3
print_job_freq(job_freq, pid)

pid = '3'
applied_jobs = find_applications(People, pid)
job_freq = find_job_freq(applied_jobs)
assert type(job_freq) is dict or type(job_freq) is defaultdict
assert len(job_freq) == 1
assert 'scientist' in job_freq and job_freq['scientist'] == 5
print_job_freq(job_freq, pid)

pid = '4'
applied_jobs = find_applications(People, pid)
job_freq = find_job_freq(applied_jobs)
assert type(job_freq) is dict or type(job_freq) is defaultdict
assert len(job_freq) == 3
assert 'scientist' in job_freq and job_freq['scientist'] == 3
assert 'data scientist' in job_freq and job_freq['data scientist'] == 1
assert 'bioinformatician' in job_freq and job_freq['bioinformatician'] == 1
print_job_freq(job_freq, pid)

print("\n(Passed!)")
```

    
    The person with pid '0' applied for 2 different types of jobs:
    1 of 'data scientist';
    1 of 'data analyst';
    
    The person with pid '1' applied for 2 different types of jobs:
    3 of 'data scientist';
    1 of 'data engineer';
    
    The person with pid '3' applied for 1 different types of jobs:
    5 of 'scientist';
    
    The person with pid '4' applied for 3 different types of jobs:
    3 of 'scientist';
    1 of 'bioinformatician';
    1 of 'data scientist';
    
    (Passed!)
    


```python
# `find_job_freq_test_2` (1 point): Test cell for person hasn't applied for any job
pid = '2'
applied_jobs = find_applications(People, pid)
job_freq = find_job_freq(applied_jobs)
assert type(job_freq) is dict or type(job_freq) is defaultdict
assert len(job_freq) == 0
print_job_freq(job_freq, pid)

print("\n(Passed!)")
```

    
    The person with pid '2' applied for 0 different types of jobs:
    
    (Passed!)
    

**Exercise 2** (3.5 points). Suppose you are given a job-frequency dictionary as might result from Exercise 1. Your task is as follows:

- Rank the jobs by frequency value in descending order.
- For jobs with the same frequency, resulting in ties, keep job titles in (ascending) alphabetical order.
- Return the final result as a list of **job titles** in the order described above.

Complete the function `rank_job(job_freq)`, below, to perform this procedure.

For example:

```python
    job_freq = {'data scientist': 1, 'bioinformatician': 1, 'scientist': 3}
    rank_job(job_freq) == ['scientist', 'bioinformatician', 'data scientist']
```

> _Hint_: Consider the _sorted_ function, per [this StackOverflow post](https://stackoverflow.com/questions/9919342/sorting-a-dictionary-by-value-then-key).
>
> _Notes_: If the input `job_freq` is an empty dictionary, then return an empty list. Also, there are several test cells, each of which checks a different condition on the output, and thereby enabling you to earn partial credit.


```python
def rank_job(job_freq):
    assert type(job_freq) is dict or type(job_freq) is defaultdict
        
    return [
        x[0]
        for x in sorted(
            job_freq.items(), key = lambda v : (-v[1], v[0])
        )
    ]
```


```python
# `rank_job_test_1` (2 points): Test cell for ranking by values correctly
applied_jobs = find_applications(People, '1')
job_freq = find_job_freq(applied_jobs)
rank = rank_job(job_freq)
assert type(rank) is list, "rank should be a list."
assert len(rank) == 2, "The Person with pid '1' only applied for 2 types of different jobs."
assert rank[0] == 'data scientist' and rank[1] == 'data engineer', "Wrong job title or wrong order."
print("\nThe applied jobs of person with pid '{}' are in this order: {}".format('1', rank))

applied_jobs = find_applications(People, '3')
job_freq = find_job_freq(applied_jobs)
rank = rank_job(job_freq)
assert type(rank) is list and len(rank) == 1
assert 'scientist' in rank
print("\nThe applied jobs of person with pid '{}' are in this order: {}".format('3', rank))

print("\n(Passed!)")
```

    
    The applied jobs of person with pid '1' are in this order: ['data scientist', 'data engineer']
    
    The applied jobs of person with pid '3' are in this order: ['scientist']
    
    (Passed!)
    


```python
# `rank_job_test_2` (1 point): Test cell for ranking by both values and keys correctly
applied_jobs = find_applications(People, '0')
job_freq = find_job_freq(applied_jobs)
rank = rank_job(job_freq)
assert type(rank) is list and len(rank) == 2
assert rank[0] == 'data analyst' and rank[1] == 'data scientist'
print("\nThe applied jobs of person with pid '{}' are in this order: {}".format('0', rank))

applied_jobs = find_applications(People, '4')
job_freq = find_job_freq(applied_jobs)
rank = rank_job(job_freq)
assert type(rank) is list and len(rank) == 3
assert rank == ['scientist', 'bioinformatician', 'data scientist']
print("\nThe applied jobs of person with pid '{}' are in this order: {}".format('4', rank))

print("\n(Passed!)")
```

    
    The applied jobs of person with pid '0' are in this order: ['data analyst', 'data scientist']
    
    The applied jobs of person with pid '4' are in this order: ['scientist', 'bioinformatician', 'data scientist']
    
    (Passed!)
    


```python
# `rank_job_test_3` (0.5 point): Test empty case
applied_jobs = find_applications(People, '2')
job_freq = find_job_freq(applied_jobs)
rank = rank_job(job_freq)
assert type(rank) is list and len(rank) == 0
print("\nThe applied jobs of person with pid '{}' are in this order: {}".format('2', rank))

print("\n(Passed!)")
```

    
    The applied jobs of person with pid '2' are in this order: []
    
    (Passed!)
    

**Exercise 3** (1.5 points). Given a job title, find all jobs from the openings database, **`'Hiring_jobs'`**, that match it. Then, return **a dictionary** with job IDs as the keys and `(title, company)` pairs as the values.

If there are no openings that match the job title, then return an empty dictionary.

There are two test cells for this exercise, to allow for partial credit.


```python
def find_top_freq_jobs(Hiring_jobs, job_title):
    
    assert type(Hiring_jobs) is dict and len(Hiring_jobs) != 0
        
    def find_top_freq_jobs__0():
        
        top_freq_jobs = {}
        
        for key, value in Hiring_jobs.items():
            
            if value[0] == job_title:
                top_freq_jobs[key] = value[0], value[1]
        
        return top_freq_jobs
    
    def find_top_freq_jobs__1():
        
        return {
            job : (pos[0], pos[1])
            for job, pos in Hiring_jobs.items()
            if pos[0] == job_title
        }
    
    return find_top_freq_jobs__1()
```


```python
# `find_top_freq_job_test_1` (1 point): Test cell
def print_top_freq_jobs(rank, top_freq_jobs, pid):
    if len(rank) >= 1:
        print("\nThe most frequently applied job for person with pid '{}' is: '{}'".format(pid, rank[0]))
        print("In Hiring_jobs, there are {} companies hiring '{}', which are:".format(len(top_freq_jobs), rank[0]))
        for _, v in top_freq_jobs.items():
            print("'{}';".format(v[1]))
    else: print("\nThe person with pid '{}' didn't apply for any jobs.".format(pid))
        
pid = '0'
applied_jobs = find_applications(People, pid)
job_freq = find_job_freq(applied_jobs)
rank = rank_job(job_freq)
top_freq_jobs = find_top_freq_jobs(Hiring_jobs, rank[0] if rank else '')
assert type(top_freq_jobs) is dict, "top_freq_jobs should be a dictionary." 
assert len(top_freq_jobs) == 1
assert '2' in top_freq_jobs and top_freq_jobs['2'] == ('data analyst', 'cnn')
print_top_freq_jobs(rank, top_freq_jobs, pid)

pid = '1'
applied_jobs = find_applications(People, pid)
job_freq = find_job_freq(applied_jobs)
rank = rank_job(job_freq)
top_freq_jobs = find_top_freq_jobs(Hiring_jobs, rank[0] if rank else '')
assert type(top_freq_jobs) is dict and len(top_freq_jobs) == 4
assert top_freq_jobs['0'] == ('data scientist', 'IBM')
assert top_freq_jobs['1'] == ('data scientist', 'coca cola')
assert top_freq_jobs['3'] == ('data scientist','google') 
assert top_freq_jobs['5'] == ('data scientist', 'apple')
print_top_freq_jobs(rank, top_freq_jobs, pid)

pid = '3'
applied_jobs = find_applications(People, pid)
job_freq = find_job_freq(applied_jobs)
rank = rank_job(job_freq)
top_freq_jobs = find_top_freq_jobs(Hiring_jobs, rank[0] if rank else '')
assert type(top_freq_jobs) is dict and len(top_freq_jobs) == 2
assert top_freq_jobs['6'] == ('scientist', 'biogen')
assert top_freq_jobs['7'] == ('scientist', 'broad institute')
print_top_freq_jobs(rank, top_freq_jobs, pid)

pid = '4'
applied_jobs = find_applications(People, pid)
job_freq = find_job_freq(applied_jobs)
rank = rank_job(job_freq)
top_freq_jobs = find_top_freq_jobs(Hiring_jobs, rank[0] if rank else '')
assert type(top_freq_jobs) is dict and len(top_freq_jobs) == 2
assert top_freq_jobs['6'] == ('scientist', 'biogen')
assert top_freq_jobs['7'] == ('scientist', 'broad institute')
print_top_freq_jobs(rank, top_freq_jobs, pid)

print("\n(Passed!)")
```

    
    The most frequently applied job for person with pid '0' is: 'data analyst'
    In Hiring_jobs, there are 1 companies hiring 'data analyst', which are:
    'cnn';
    
    The most frequently applied job for person with pid '1' is: 'data scientist'
    In Hiring_jobs, there are 4 companies hiring 'data scientist', which are:
    'IBM';
    'coca cola';
    'google';
    'apple';
    
    The most frequently applied job for person with pid '3' is: 'scientist'
    In Hiring_jobs, there are 2 companies hiring 'scientist', which are:
    'biogen';
    'broad institute';
    
    The most frequently applied job for person with pid '4' is: 'scientist'
    In Hiring_jobs, there are 2 companies hiring 'scientist', which are:
    'biogen';
    'broad institute';
    
    (Passed!)
    


```python
# `find_top_freq_job_test_2` (0.5 point): Test cell 
pid = '2'
applied_jobs = find_applications(People, pid)
job_freq = find_job_freq(applied_jobs)
rank = rank_job(job_freq)
top_freq_jobs = find_top_freq_jobs(Hiring_jobs, rank[0] if rank else '')
assert type(top_freq_jobs) is dict and len(top_freq_jobs) == 0
print_top_freq_jobs(rank, top_freq_jobs, pid)

print("\n(Passed!)")
```

    
    The person with pid '2' didn't apply for any jobs.
    
    (Passed!)
    

**Exercise 4** (2 points). Now let's put all these pieces together into a "job recommender," using the following simple scheme based on Exercises 0-3.

In particular, suppose you are given a person ID, `pid`, the `People` dictionary, and the `Hiring_jobs` dictionary. You can then use the previous functions to do the following.

- Determine what job titles the person applied for, and how often. (Exercises 0 and 1)
- Rank these titles in descending order of application frequency. (Exercise 2)
- Determine all available openings whose titles match the person's top-ranked job title. (Exercise 3)

Then, your job recommender can scan the openings in `Hiring_jobs` for this job title and return them.

Implement this procedure in `recommend_jobs(People, pid, Hiring_jobs)`, below. It should return a **dictionary**, whose keys are the job IDs (from `Hiring_jobs`) and the values are `(title, company)` pairs. **However**, you must also **filter out jobs the person has already applied for.** For example, consider Person `'1'`:


```python
print(People['1'])
```

    {'0': {'title': 'data scientist', 'company': 'google', 'status': 'apply'}, '1': {'title': 'data engineer', 'company': 'amazon', 'status': 'apply'}, '2': {'title': 'data scientist', 'company': 'facebook', 'status': 'apply'}, '3': {'title': 'data scientist', 'company': 'microsoft', 'status': 'apply'}, '4': {'title': 'data analyst', 'company': 'chase', 'status': 'previous'}}
    

Person `'1'` already applied for the `'data scientist'` opening at Google. In the `Hiring_jobs` database, this position has a job ID of `'3'`:


```python
print(Hiring_jobs)
```

    {'0': ['data scientist', 'IBM'], '1': ['data scientist', 'coca cola'], '2': ['data analyst', 'cnn'], '3': ['data scientist', 'google'], '4': ['business analyst', 'chase'], '5': ['data scientist', 'apple'], '6': ['scientist', 'biogen'], '7': ['scientist', 'broad institute'], '8': ['postdoc', 'mit'], '9': ['data engineer', 'microsoft']}
    

Now, suppose `data scientist` turns out to be the job title to which Person 1 applied most often. Then, `{'3':('data scientist', 'google')}` should never appear in his recommended job list. Instead, only jobs '0','1' and '5' from `Hiring_jobs` should be recommended.

> **Note.** If a person hasn't applied for any job, then you shouldn't recommend any job to him (your function should return an empty dictionary).

This procedure is, of course, a very simple one. In a more realistic application, you would want to to check if that the job is relevant to the candidate's profile and consider many other factors before making a job recommendation to a user.

Enter your proposed solution, below. There are two test cells for this exercise so you can try to get partial credit.


```python
def recommend_jobs(People, pid, Hiring_jobs):
    assert type(People) is dict and len(People) != 0
    assert type(pid) is str
    assert type(Hiring_jobs) is dict and len(Hiring_jobs) != 0
    
    applied_jobs = find_applications(People, pid)
    job_freq = find_job_freq(applied_jobs)
    rank = rank_job(job_freq)
    top_freq_jobs = find_top_freq_jobs(Hiring_jobs, rank[0] if rank else '')
    
    recommend_jobs = {}
    if top_freq_jobs:        
        for key, value in top_freq_jobs.items():
            if value not in applied_jobs:
                recommend_jobs[key] = value
        
    return recommend_jobs
```


```python
# `recommend_jobs_test_1` (1 point): Test cell for general cases
def print_recommends(recommends, pid):
    assert type(recommends) is dict
    if len(recommends) >= 1:
        print("\nThe recommended jobs for person with pid '{}' are:".format(pid))
        for k, v in recommends.items():
            print("'{}':{};".format(k, v))
    else: print("\nThere is no recommended job for person with pid '{}'.".format(pid))


pid = '0'
recommends = recommend_jobs(People, pid, Hiring_jobs)
assert type(recommends) is dict, "recommends should be a dict."
assert len(recommends) == 1
assert recommends['2'] == ('data analyst', 'cnn'), "Wrong recommendation."
print_recommends(recommends, pid)
              

pid = '3'
recommends = recommend_jobs(People, pid, Hiring_jobs)
assert type(recommends) is dict and len(recommends) == 2
assert recommends['6'] == ('scientist', 'biogen')
assert recommends['7'] == ('scientist', 'broad institute')
print_recommends(recommends, pid)

pid = '4'
recommends = recommend_jobs(People, pid, Hiring_jobs)
assert type(recommends) is dict and len(recommends) == 2
assert recommends['6'] == ('scientist', 'biogen')
assert recommends['7'] == ('scientist', 'broad institute')
print_recommends(recommends, pid)

print("\n(Passed!)")
```

    
    The recommended jobs for person with pid '0' are:
    '2':('data analyst', 'cnn');
    
    The recommended jobs for person with pid '3' are:
    '6':('scientist', 'biogen');
    '7':('scientist', 'broad institute');
    
    The recommended jobs for person with pid '4' are:
    '6':('scientist', 'biogen');
    '7':('scientist', 'broad institute');
    
    (Passed!)
    


```python
# `recommend_jobs_test_2` (1 point): Test cell for special cases

# test whether you successfully filter out the applied jobs
pid = '1'
recommends = recommend_jobs(People, pid, Hiring_jobs)
assert type(recommends) is dict and len(recommends) == 3
assert recommends['0'] == ('data scientist', 'IBM')
assert recommends['1'] == ('data scientist', 'coca cola')
assert recommends['5'] == ('data scientist', 'apple')
print_recommends(recommends, pid)

# test that you don't recommend jobs to the person who hasn't applied any job
pid = '2'
recommends = recommend_jobs(People, pid, Hiring_jobs)
assert type(recommends) is dict and len(recommends) == 0
print_recommends(recommends, pid)

print("\n(Passed!)")
```

    
    The recommended jobs for person with pid '1' are:
    '0':('data scientist', 'IBM');
    '1':('data scientist', 'coca cola');
    '5':('data scientist', 'apple');
    
    There is no recommended job for person with pid '2'.
    
    (Passed!)
    

**Fin!** You've reached the end of this problem. Don't forget to restart the 
kernel and run the entire notebook from top-to-bottom to make sure you did 
everything correctly. If that is working, try submitting this problem. 
(Recall that you *must* submit and pass the autograder to get credit for your work!)
