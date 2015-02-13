# Contributing to LSTM
We want to make contributing to this project as easy and transparent as
possible.

## Our Development Process
This project is developed internally at Facebook inside a private repository.
Changes are periodically pushed to the open-source branch. Pull requests are
integrated manually into our private repository first, and they then get
propagated to the public repository with the next push.

## Pull Requests
We actively welcome your pull requests.
1. Fork the repo and create your branch from `master`.
2. If you've added code that should be tested, add tests
3. Make sure your code lints.
4. If you haven't already, complete the Contributor License Agreement ("CLA").

## Contributor License Agreement ("CLA")
In order to accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Facebook's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

## Issues
We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

Facebook has a [bounty program](https://www.facebook.com/whitehat/) for the safe
disclosure of security bugs. In those cases, please go through the process
outlined on that page and do not file a public issue.

## Coding Style

### C++
* 2 spaces for indentation rather than tabs
* 80 character line length
* Name classes LikeThis, functions and methods likeThis, data members
likeThis_.
* Most naming and formatting recommendations from
[Google's C++ Coding Style Guide](
http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml) apply (but
not the restrictions; exceptions and templates are fine.)
* Feel free to use [boost](http://www.boost.org/),
[folly](https://github.com/facebook/folly) and
[fbthrift](https://github.com/facebook/fbthrift)

### Lua
* Inspired by [PEP 8](http://legacy.python.org/dev/peps/pep-0008/)
* 4 spaces for indentation rather than tabs
* 80 character line length
* Name classes LikeThis, functions, methods, and variables like_this, private
methods _like_this
* Use [Penlight](http://stevedonovan.github.io/Penlight/api/index.html);
specifically pl.class for OOP
* Do not use global variables (except with a very good reason)
* Use [new-style modules](http://lua-users.org/wiki/ModulesTutorial); do not
use the module() function
* Assume [LuaJIT 2.0+](http://luajit.org/), so Lua 5.1 code with LuaJIT's
supported [extensions](http://luajit.org/extensions.html);
[FFI](http://luajit.org/ext_ffi.html) is okay.

## License
By contributing to LSTM, you agree that your contributions will be licensed
under its Apache 2.
