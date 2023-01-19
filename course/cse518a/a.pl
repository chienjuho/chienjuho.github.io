#!/usr/bin/perl

system "mv ~/Downloads/CourseSchedule.html ./";
system "./remove-redirect.pl CourseSchedule.html > schedule.html";
