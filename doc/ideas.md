# Personalized Simple All-to-All Algorithms

-   Current Findings
    -   Flat Handshake is the most efficient with large message sizes. We still
          have to find out with a large number of cores
    -   All-to-All is most efficient with very small messages and a large node
          count
-   Ideas
    -   Instead of blocking _dart_sendrecv_ let us use non blocking send-receive
          to overlap communication with multiple partners in multiple phases.
    -   Really exploit the leader model
