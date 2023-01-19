#!/usr/bin/perl

open my $FH, $ARGV[0];

#href="https://www.google.com/url?q=https://www.wired.com/2006/06/crowds/&amp;sa=D&amp;ust=1547516307507000"
#href="https://www.google.com/url?q=https://www.wired.com/2006/06/crowds/&amp;sa=D&amp;ust=1599949237720000&amp;usg=AOvVaw3YKK1kD9hYF5d4lgBZK-uZ"
#https://www.google.com/url?q=https://www.wired.com/2006/06/crowds/&amp;sa=D&amp;source=editors&amp;ust=1630030032126000&amp;usg=AOvVaw1s3kSLS0xxYXcYrRhckyuB
while (<$FH>){
    chomp;
    #s/href="https:\/\/www\.google\.com\/url?q=/href="/g;
    s/href="https:\/\/www\.google\.com\/url\?q=/href="/g;
    s#&amp;sa=D&amp;[^"]+"#"#g;
    s/dl\.acm\.org\/citation\.cfm\?id%3D/dl.acm.org\/citation.cfm?id=/g;
    s/<html>/<html><base target="_blank">/g;
    print "$_";
}

close $FH;
