#!/usr/bin/perl -w

use strict;
use warnings;

use Text::Diff;
use File::stat;
use List::Util qw(min);

my %include_list;
my %include_map;

sub process_cpp {
    my ($path) = @_;

    # read file
    open( F, $path ) or die("Cannot read file $path: $!");
    my @data = <F>;
    close(F);

    # put all #include lines into the includelist+map
    for my $i ( 0 ... @data - 1 ) {
        if ( $data[$i] =~ m!\s*#\s*include\s*[<"](\S+)[">]! ) {
            push( @{ $include_list{$1} }, $path );
            $include_map{$path}{$1} = 1;
        }
    }
}

use Git;

my $repo     = Git->repository();
my @filelist = $repo->command('ls-files');

foreach my $file (@filelist) {
    if ( $file =~ /\.(h|cpp|hpp|h\.in|dox)$/ ) {
        process_cpp($file);
    }
}

my %graph = %include_map;

# Tarjan's Strongly Connected Components Algorithm
# https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm

my $index = 0;
my @S     = [];
my %vi;    # vertex info

sub strongconnect {
    my ($v) = @_;

    # Set the depth index for v to the smallest unused index
    $vi{$v}{index}   = $index;
    $vi{$v}{lowlink} = $index++;
    push( @S, $v );
    $vi{$v}{onStack} = 1;

    # Consider successors of v
    foreach my $w ( keys %{ $graph{$v} } ) {
        if ( !defined $vi{$w}{index} ) {

            # Successor w has not yet been visited; recurse on it
            strongconnect($w);
            $vi{$w}{lowlink} = min( $vi{$v}{lowlink}, $vi{$w}{lowlink} );
        }
        elsif ( $vi{$w}{onStack} ) {

            # Successor w is in stack S and hence in the current SCC
            $vi{$v}{lowlink} = min( $vi{$v}{lowlink}, $vi{$w}{index} );
        }
    }

    # If v is a root node, pop the stack and generate an SCC
    if ( $vi{$v}{lowlink} == $vi{$v}{index} ) {

        # start a new strongly connected component
        my @SCC;
        my $w;
        do {
            $w = pop(@S);
            $vi{$w}{onStack} = 0;

            # add w to current strongly connected component
            push( @SCC, $w );
        } while ( $w ne $v );

        # output the current strongly connected component (only if it is not
        # a singleton)
        if ( @SCC != 1 ) {
            print "-------------------------------------------------";
            print "Found cycle:\n";
            print "    $_\n" foreach @SCC;
            print "end cycle\n";
        }
    }
}

foreach my $v ( keys %graph ) {
    next if defined $vi{$v}{index};
    strongconnect($v);
}

print "Writing include_map:\n";
foreach my $inc ( sort keys %include_map ) {
    print "$inc => " . scalar( keys %{ $include_map{$inc} } ) . " [";
    print join( ",", sort keys %{ $include_map{$inc} } ) . "]\n";
}
