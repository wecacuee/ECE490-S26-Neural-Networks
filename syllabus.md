# **ECE 490/590: Neural Networks (Spring 2026\)**

## **Course Information**

This course introduces the fundamentals of artificial neural networks. It provides lessons on how neural networks work. The course starts from explaining a single neuron, building to a layer of neurons, and then a multi-layer neural network. Apart from basics of building neural networks using libraries like PyTorch, we dive into details of how PyTorch works behind the scenes, including automatic differentiation (both forward and reverse mode) and stochastic optimization.

ECE 590 requires additional demonstration of the ability to read, understand, and review academic literature in neural networks over the requirements of ECE 490\.

ECE 590 and ECE 490 cannot both be taken for degree credits.

Number of credit hours: 3

### **Prerequisites:**

(MAT 258 or MAT 262\) AND (ECE 277 or COS 225\) OR Instructor Permission

Explanation: The course requires a background in Linear algebra, and programming. The following three courses span the prerequisites:

1. Linear Algebra: (MAT 258 or MAT 262\) AND,  
2. Python object oriented Programming: (ECE 277 or COS 225\)

### **Co-requisite:** 

STS 235 OR Instructor Permission

Explanation: The course requires a background in Probability and statistics

## **Course Delivery Method**

### **Mode of Instruction** 

Asynchronous online

### **Digital Services, Hardware, Software** 

Learning Management System is. Brightspace, Web or Video Conferencing Service is Zoom, Video Recording/Sharing Service is Kaltura, Collaboration and Communication Services is Google Drive and Docs. 

Students are expected to work on their own computer / laptop. Programming assignments and projects will be developed in the Python programming language. We will also use the Pytorch deep learning library for some homeworks and for the project. Students are expected to use free Google CoLab and the Advanced Computing Group (AGC) if necessary.

## **Faculty Information** 

Dr. Vikas Dhiman, Rm 279, 5708 Barrows Hall,  Email: [vikas.dhiman@maine.edu](mailto:vikas.dhiman@maine.edu) ,   
Office hours will be conducted in-person or on Zoom: [https://maine.zoom.us/my/vikas.dhiman](https://maine.zoom.us/my/vikas.dhiman)   
**Weekly Code Jam sessions: 3:30-4:30 pm on Fridays**  
Course website: [vikasdhiman.info/ECE490-F25-Neural-Networks](https://vikasdhiman.info/ECE490-F25-Neural-Networks)

## **Instructional Materials and Methods** 

### **Textbooks**

There are two main textbooks, both openly available online:

* Mathematics for Machine Learning. Marc Peter Deisenroth A. Aldo Faisal Cheng Soon Ong [https://mml-book.github.io/book/mml-book.pdf](https://mml-book.github.io/book/mml-book.pdf)   
* Understanding Deep Learning \- Simon J.D. Prince [https://github.com/udlbook/udlbook/releases/download/v.1.18/UnderstandingDeepLearning\_24\_12\_23\_C.pdf](https://github.com/udlbook/udlbook/releases/download/v.1.18/UnderstandingDeepLearning_24_12_23_C.pdf) 

Supplementary material:

* PATTERNS, PREDICTIONS, AND ACTIONS by Moritz Hardt and Benjamin Recht [https://mlstory.org/](https://mlstory.org/)   
* Deep Learning by Ian Goodfellow, Yoshua Bengio, Aaron Courville. (Available here: [https://www.deeplearningbook.org/](https://www.deeplearningbook.org/)  )  
* Machine Learning: A Probabilistic Perspective by Kevin P. Murphy (Available here: [https://github.com/probml/pml-book/releases/latest/download/book1.pdf](https://github.com/probml/pml-book/releases/latest/download/book1.pdf) )  
* Pattern Recognition and Machine Learning, Christopher Bishop, (Available here: [https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning) )

## **Course Goals:** 

1. Students should learn about the fundamentals of Neural Networks (NN) and Machine Learning algorithms.  
2. Students should be able to comprehend the strengths and weaknesses of Neural Networks as compared to other Machine Learning methods.  
3. Students should be able to solve real world problems using Neural Networks.

### **Instructional Objectives:** 

After completing this course, the students will be able to

1. Understand and implement auto-differentiation in Python  
2. Understand and analyze the parts of a machine learning algorithm, including data preprocessing, initialization, loss function, models and optimization algorithm.  
3. Understand the strengths and limitations of different components of neural networks to pick the relevant components to solve new problems.

### **Student Learning Outcomes** 

Upon completion of the course, students will be able to:

* SLO 1: Use open-source software for neural networks  
* SLO 2: Understand foundations, limitations and strengths of neural networks  
* SLO 3: Ability to understand and debug neural networks on different problems and datasets

### **Grading and Course Expectations** 

Homeworks will be due every other week.

| ECE 490 |  | ECE 590 |  |
| :---- | :---- | :---- | :---- |
| Exams | 30% | Exams | 30% |
| Assignments | 35% | Assignments | 35% |
| Project | 35% | Project | 25% |
| Class participation | 5% | Class participation | 5% |
|  |  | Paper review and presentation | 10% |

Grading will be based on absolute scale 90% (A)/80% (B)/70% (C)/60% (D).

#### **Paper review and presentation**

The graduate students will be asked to review a recent neural network paper of their choice, write a 2-page review and record a 15-min presentation video. 

## **Tentative Course Schedule** 

| Date | Topic | Key concepts | Assessment |
| ----- | ----- | ----- | ----- |
| 01/20 | [Introduction to Neural Networks](https://vikasdhiman.info/ECE490-F25-Neural-Networks/notebooks/intro/what-to-expect/) |  | [HW0](https://vikasdhiman.info/ECE490-F25-Neural-Networks/notebooks/intro/prereq-hw/) |
| 01/22 | [Running Jupyter on PSC Cluster](https://vikasdhiman.info/ECE490-F25-Neural-Networks/tutorials/acg-slurm-jupyter/) |  |  |
| 01/27 | [Introduction to Python 1.ipynb](https://vikasdhiman.info/ECE490-F25-Neural-Networks/notebooks/py-intro/python-1/) |  |  |
| 01/29 | [High level tour of Neural Networks](https://vikasdhiman.info/ECE490-F25-Neural-Networks/tutorials/tip-of-the-iceberg/) |  | Python quiz |
| 02/03 | [Linear Algebra Review.pdf](https://github.com/wecacuee/ECE490-F25-Neural-Networks/blob/master/notebooks/02-linear-models/exports/20250916-linear-models.pdf.pdf) |  | Discussion |
| 02/05 | [Linear Regression](https://github.com/wecacuee/ECE490-F25-Neural-Networks/blob/master/notebooks/02-linear-models/exports/20250918-linear-models.pdf.pdf.pdf) |  | HW2 |
| 02/10 | [Linear Regression](https://github.com/wecacuee/ECE490-F25-Neural-Networks/blob/master/notebooks/02-linear-models/exports/20250923-linear-models.pdf.pdf.pdf) |  |  |
| 02/12 | [Continuous Optimization](https://github.com/wecacuee/ECE490-F25-Neural-Networks/blob/master/notebooks/02-linear-models/exports/20250925-cont-opt.pdf.pdf) |  | Linear Algebra Quiz |
| 02/17 | [Perceptron3](https://github.com/wecacuee/ECE490-F25-Neural-Networks/blob/master/notebooks/02-linear-models/exports/20250930-40-perceptron.pdf.pdf) |  |  |
| 02/19 | [Probabilistic Perspective](https://github.com/wecacuee/ECE490-F25-Neural-Networks/blob/master/notebooks/022-prob/2025-10-02-SGD.pdf) |  | [HW3](https://colab.research.google.com/github/wecacuee/ECE490-F25-Neural-Networks/blob/master/notebooks/02-linear-models/exports/40-Perceptron3%20Colab.ipynb) |
| 02/24 | [Probabilistic Perspective](https://github.com/wecacuee/ECE490-F25-Neural-Networks/blob/master/notebooks/022-prob/exports/2025-10-07-ProbabilisticPerspective.pdf.pdf) |  |  |
| 02/26 | Midterm 1 |  |  |
| 03/03 | Fall Break: no class |  |  |
| 03/05 | Midterm Solution |  | Discussion |
| 03/10 | [Optimization algorithms](https://github.com/wecacuee/ECE490-F25-Neural-Networks/blob/master/notebooks/06-pytorch/20251021-UDLBook-Prince-p93-104.pdf.pdf) |  | Optimization basics |
| 03/12 | [MLP.pdf.pdf](https://github.com/wecacuee/ECE490-F25-Neural-Networks/blob/master/notebooks/05-mlp/2025-10-23-MLP.pdf.pdf) |  |  |
| 03/17 | [CrossEntropyLoss-WeightDecay.pdf](https://github.com/wecacuee/ECE490-F25-Neural-Networks/blob/master/notebooks/06-pytorch/exports/2025-10-28-CrossEntropyLoss-WeightDecay.pdf) |  | MLP quiz |
| 03/19 | No class |  |  |
| 03/24 | [Autograd.pdf](https://github.com/wecacuee/ECE490-F25-Neural-Networks/blob/master/notebooks/03-autograd/Autograd.pdf.pdf) |  | Discussion |
| 03/26 | Midterm 2 |  |  |
| 03/31 | Veterans Day: no class |  |  |
| 04/02 | [Autograd.pdf](https://github.com/wecacuee/ECE490-F25-Neural-Networks/blob/master/notebooks/03-autograd/2025-11-13-Autograd.pdf.pdf) |  |  |
| 04/07 | [Autograd.pdf](https://github.com/wecacuee/ECE490-F25-Neural-Networks/blob/master/notebooks/03-autograd/2025-11-18-Autograd.pdf.pdf) |  |  |
| 04/09 | [AutogradNumpy.pdf](https://github.com/wecacuee/ECE490-F25-Neural-Networks/blob/master/notebooks/03-autograd/2025-11-20-AutogradNumpy.pdf.pdf) |  |  |
| 04/14 | [AutogradNumpy.pdf](https://github.com/wecacuee/ECE490-F25-Neural-Networks/blob/master/notebooks/03-autograd/2025-11-25-AutogradNumpy.pdf.pdf) |  | Autograd quiz |
| 04/16 | Thanks giving: no class |  |  |
| 04/21 | Snow day: no class |  |  |
| 04/23 | Architectures |  | Discussion |
| 04/28 | Presentations |  |  |
| 04/30 | Presentations |  | [Review](https://vikasdhiman.info/ECE490-F25-Neural-Networks/notebooks/review/review/) |

## **Course Policies** 

* Late submission will be penalized with a 10% late penalty and no submissions are allowed after grading and solutions are released. You can use one wild-card for late submissions, use it carefully: people get sick, laptops get broken, accidents happen.  
* There will be no make-ups for missed exams; plan to be there.   
* Learning from external sources (books, API documentation, YouTube videos, internet blogs) is encouraged. You should seek out the information you need. Seeking information that you need and learning it from other resources is a skill that needs to be developed. However, use of external sources to solve assignments or copy projects is discouraged. Get parts of code from the internet, AI tools like ChatGPT is tolerated as long as you clearly demarcate your work from external sources. Also be aware that ChatGPT can often be wrong. Also, attribute/cite/refer the work to the proper source.

#### **Collaboration policy**

You are encouraged to discuss concepts, compare approaches, and study in groups to build understanding. However, when it comes time to create your own written homework solutions, you must work independently and write in your own words.

 ✅ Allowed: talking through example problems together, comparing notes, sharing strategies, reviewing definitions.  
 ❌ Not allowed: copying someone else’s written work, dividing up problems so each person writes only part, or submitting shared text.

If you are unsure whether a type of collaboration is okay, please ask me in advance. My goal is to support your learning and ensure that what you turn in reflects your own understanding.

#### **External sources**

You are encouraged to look up information—books, API documentation, forums, videos, blogs, and tools like ChatGPT can all be valuable for learning. Part of this course is developing the skill of finding and understanding resources.

✅ Allowed and encouraged:

– Reading documentation or watching videos to understand a concept.

– Looking up small code snippets (e.g., syntax examples, library calls) and adapting them.

– Using AI tools (like ChatGPT) to explain a topic or generate starter examples as long as you rewrite, test, and understand the code yourself.

❌ Not allowed:

– Submitting code or solutions copied directly from any external source.

– Using AI or internet code to complete whole assignments without modification or understanding.

– Dividing a project so someone else completes major sections for you.

When you use external sources:

– Clearly mark which parts are influenced by external material (e.g., in comments: \# adapted from \[source\]).

– Provide a short citation in your submission (a link, book title, or “ChatGPT conversation on X date”).

– Remember: AI tools can be wrong or misleading—always test and verify.

If you are unsure whether your use of an external source is appropriate, ask me before submitting. The goal is to support your learning while ensuring your final submission reflects your own understanding.

### **Getting Started for the course**

* Introduce yourself on Brightspace discussion forum “Introduction thread”

### **Support expectations from the instructor**

* Response time: I will respond to emails within 48 hours on weekdays.   
* Grading timeline: Assignments will be graded within 7-14 days of submission.

### **Success Strategies**

* Review lectures. Follow corresponding sections of the textbook MML or UDLBook.

### **Tutoring options**

Tutoring is available for a wide range of UMaine courses using the tutor matching platform Knack. To meet with a UMaine peer tutor, go to [umaine.joinknack.com](http://umaine.joinknack.com/) and sign up with your MaineStreet account. During signup, there will also be an opportunity to download the Knack app for [apple](https://knack.to/ios) or [android](https://knack.to/android). Using the Knack app, you will be able to select the course in which you want tutoring and the tutors will receive a notification of your request. Communication with the tutor is done through the Knack app to arrange a time and location for your tutoring session. Questions? Contact: Office of Student Academic Success in 104 Dunn Hall, call 207-581-2351 or send an email to [um.osas@maine.edu](mailto:um.osas@maine.edu)  

**How much does tutoring cost?** 

Tutoring is completely free to UMaine students.

**Who are the tutors?** 

Tutors are UMaine students—peers who have earned a B+ or better in the course. They are paid, trained tutors, and students seeking tutoring can choose their tutor from a list of available UMaine peer tutors.

### **ChatGPT and other Generative AI policy**

Similar to using search engines, how you use Gen-AI is what matters. You should use search engines rather than generative AI because you can see the source of the information.

Similar to search engines, it is not acceptable to ask for solutions to homework problems from Gen-AI. Gen-AI might be able to solve homework problems because the solutions are typically available on the internet. However, Gen-AI is not yet smart enough to solve real-life problems.

It is acceptable to use code autocompletion tools like GitHub Copilot. It is also acceptable to use Grammarly for correcting language. It is NOT acceptable to start with the initial draft provided by Gen-AI. This is due to several reasons.

1. **Plagiarism and falsehoods:** AI is trained on a vast amount of written text from the internet, books, and papers. It has been aptly described as stochastic parrots by Bender et al. (2021). I would rather call AI stochastic plagiarizers. In an AI-written primary draft, the likelihood of sentences appearing verbatim from unknown sources is too high for my comfort. We can run the draft through "Turnitin"-like tools, but that seems like cheating as well. It is similar to plagiarism followed by paraphrasing until it passes the "Turnitin" test. For the same reasons, the probability of the draft containing false statements is too high for my comfort level. Furthermore, when you cite a statement from a paper, you are expected to have read the paper (or at least skimmed it). Citing a paper because AI suggested it, without having read it, is dishonest and a minefield of punishable dishonesty.  
2. **Disrespect to readers, reviewers, and the written word:** Graders, reviewers and readers assume that the papers they are reading are written by humans. Even if we disclose that the paper was written by an AI, why can't the graders/reviewers let the AI read the paper and make a decision? Are we reaching a world where AI writes for other AIs? I don't like that world. If we don't have ideas to write 18 pages ourselves and are expanding 4 pages' worth of bullet points into 18 pages that get summarized by AI back to 4 pages, maybe we should not be communicating in 18 pages at all. We should be sending 4 pages of bullet points that were inputted to an AI.

Using an AI-written draft for a report or homework is a symptom of disrespect for the written word. Yes, we are surrounded by superfluous writing and bad writing, but that does not mean we should fill the world with AI-written meaningless drivel. Instead, we should look back and fix the systems that require meaningless paperwork from humans. The alternative solution to meaningless paperwork might be exchanging bullet point lists instead of introducing a layer of AI writing and then AI summarization in between human communication.

### **Campus Policies** 

#### **Academic Honesty Statement:** 

Academic honesty is very important. It is dishonest to cheat on exams, to copy term papers, to submit papers written by another person, to fake experimental results, or to copy or reword parts of books or articles into your own papers without appropriately citing the source. Students committing or aiding in any of these violations may be given failing grades for an assignment or for an entire course, at the discretion of the instructor. In addition to any academic action taken by an instructor, these violations are also subject to action under the University of Maine Student Conduct Code.  The maximum possible sanction under the student conduct code is dismissal from the University.

Please see the University of Maine System's Academic Integrity Policy  listed in the Board Policy Manual as Policy 314: [https://www.maine.edu/board-of-trustees/policy-manual/section-314/](https://www.maine.edu/board-of-trustees/policy-manual/section-314/)

#### **Students Accessibility Services Statement** 

If you have a disability for which you may be requesting an accommodation, please contact Student Accessibility Services, located at the Center for Accessibility and Volunteer Engagement at the UCU, 139 Rangeley Rd, um.sas@maine.edu, 207.581.2319, as early as possible in the term. Students may begin the accommodation process by submitting an accommodation request form online and uploading documentation at [https://umaine-accommodate.symplicity.com/public\_accommodation/](https://umaine-accommodate.symplicity.com/public_accommodation/). Once students meet with SAS and eligibility has been determined, students submit an online request with SAS each semester to activate their approved accommodations. SAS creates an accessibility letter each semester which informs faculty of potential course access and approved reasonable accommodations; the letter is sent directly to the course instructor. Students who have already been approved for accommodations by SAS and have a current accommodation letter should meet with me (Vikas Dhiman \<vikas.dhiman@maine.edu\> Barrows 279\) privately as soon as possible.

#### **Course Schedule Disclaimer (Disruption Clause):** 

In the event of an extended disruption of normal classroom activities (due to COVID-19 or other long-term disruptions), the format for this course may be modified to enable its completion within its programmed time frame. In that event, you will be provided an addendum to the syllabus that will supersede this version.

#### **Observance of Religious Holidays/Events:** 

The University of Maine recognizes that when students are observing significant religious holidays, some may be unable to attend classes or labs, study, take tests, or work on other assignments. If they provide adequate notice (at least one week and longer if at all possible), these students are allowed to make up course requirements as long as this effort does not create an unreasonable burden upon the instructor, department or University. At the discretion of the instructor, such coursework could be due before or after the examination or assignment. No adverse or prejudicial effects shall result to a student’s grade for the examination, study, or course requirement on the day of religious observance. The student shall not be marked absent from the class due to observing a significant religious holiday. In the case of an internship or clinical, students should refer to the applicable policy in place by the employer or site. 

#### **Sexual Violence Policy**  

##### **Sexual Discrimination Reporting**

The University of Maine is committed to making campus a safe place for students. Because of this commitment, if you tell a faculty or staff member who is deemed a “responsible employee” about sexual discrimination, they are required to report this information to Title IX Student Services or the Office of Equal Opportunity.

Behaviors that can be “sexual discrimination” include sexual assault, sexual harassment, stalking, relationship abuse (dating violence and domestic violence), sexual misconduct, and gender discrimination.  Therefore, all of these behaviors must be reported.

 **Why do teachers have to report sexual discrimination?**

The University can better support students in trouble if we know about what is happening. Reporting also helps us to identify patterns that might arise – for example, if more than one person reports having been assaulted or harassed by the same individual.

 **What will happen to a student if a teacher reports?**

An employee from Title IX Student Services or the Office of Equal Opportunity will reach out to you and offer support, resources, and information.  You will be invited to meet with the employee to discuss the situation and the various options available to you.

If you have requested confidentiality, the University will weigh your request that no action be taken against the institution’s obligation to provide a safe, nondiscriminatory environment for all students.  If the University determines that it can maintain confidentiality, you must understand that the institution’s ability to meaningfully investigate the incident and pursue disciplinary action, if warranted, may be limited.  There are times when the University may not be able to honor a request for confidentiality because doing so would pose a risk to its ability to provide a safe, nondiscriminatory environment for everyone. If the University determines that it cannot maintain confidentiality, the University will advise you, prior to starting an investigation and, to the extent possible, will share information only with those responsible for handling the institution’s response

The University is committed to the well-being of all students and will take steps to protect all involved from retaliation or harm.

**If you want to talk** **in confidence** to someone about an experience of sexual discrimination, please contact these resources:

For *confidential resources on campus*: **Counseling Center: 207-581-1392** or **Cutler Health Center: at 207-581-4000**.

For *confidential resources off campus*:  **Rape Response Services:** 1-800-871-7741 or **Partners for Peace**: 1-800-863-9909.

**Other resources:**  The resources listed below can offer support but may have to report the incident to others who can help:

For *support services on campus*: **Title IX Student Services: 207-581-1406**, **Office of Community Standards: 207-581-1406**, **University of Maine Police: 207-581-4040 or 911**. 

[Visit the Title IX Student Services website at umaine.edu/titleix/ for more information.](https://umaine.edu/titleix/)

For the latest language on campus policies please refer to [https://umaine.edu/citl/teaching-resources-2/required-syllabus-information/](https://umaine.edu/citl/teaching-resources-2/required-syllabus-information/)

