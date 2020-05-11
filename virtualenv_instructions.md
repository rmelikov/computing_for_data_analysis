## virtualenv instructions

To create a virtual development environment for a project with a specific version of Python, you must use `virtualenv` package (`pip install virtualenv`).

After that, navigate to the project folder or, if you prefer to keep environments in a separate location, navigate to the environments location. However, some experts suggest to have projects have their own environments so that when sharing a project the environment goes with it.

```
> cd D:\git\CSE6040
D:\git\CSE6040>virtualenv -p "C:\Program Files\Python37\python.exe" venv37
D:\git\CSE6040>cd venv37/Scripts
D:\git\CSE6040\venv37\Scripts>activate
```

After you compete your development, you should probably save all of the dependencies requirements into a file like so:

```
(venv37) D:\git\CSE6040>pip freeze --local > requirements.txt
```

And finally, when you're done, you should deactivate the environement like so:

```
(venv37) D:\git\CSE6040> cd venv37/Scripts
(venv37) D:\git\CSE6040\venv37\Scripts>deactivate
```
