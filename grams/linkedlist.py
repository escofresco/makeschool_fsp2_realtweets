#!python


class Node(object):
    def __init__(self, data):
        """Initialize this node with the given data."""
        self.data = data
        self.next = None

    def __repr__(self):
        """Return a string representation of this node."""
        return 'Node({!r})'.format(self.data)


class LinkedList(object):
    def __init__(self, items=None):
        """Initialize this linked list and append the given items, if any."""
        self.head = None  # First node
        self.tail = None  # Last node
        self.list_length = 0  # track length of this list
        # Append given items
        if items is not None:
            for item in items:
                self.append(item)

    def __str__(self):
        """Return a formatted string representation of this linked list."""
        items = ['({!r})'.format(item) for item in self.items()]
        return '[{}]'.format(' -> '.join(items))

    def __repr__(self):
        """Return a string representation of this linked list."""
        return 'LinkedList({!r})'.format(self.items())

    def items(self):
        """Return a list (dynamic array) of all items in this linked list.
        time: Ω(n) = O(n) -> Θ(n)   we always need to loop through all n nodes to
                                    get each item.
        """
        items = []  # O(1) time to create empty list
        # Start at head node
        node = self.head  # O(1) time to assign new variable
        # Loop until node is None, which is one node too far past tail
        while node is not None:  # Always n iterations because no early return
            items.append(node.data)  # O(1) time (on average) to append to list
            # Skip to next node to advance forward in linked list
            node = node.next  # O(1) time to reassign variable
        # Now list contains items from all nodes
        return items  # O(1) time to return list

    def is_empty(self):
        """Return a boolean indicating whether this linked list is empty.
        time: Ω(1) = O(1) -> Θ(1) the list is empty when head is None
        """
        return self.head is None

    def length(self):
        """Return the length of this linked list by traversing its nodes.
        time: Ω(1) = O(1) -> Θ(1) length is tracked through append and delete.
        """
        return self.list_length

    def append(self, item):
        """Insert the given item at the tail of this linked list.
        time:Ω(1) = O(1) -> Θ(1) all we have to do is set next pointer of tail.
        """
        cur = Node(item)
        if self.tail is None:
            self.head = self.tail = cur
        else:
            self.tail.next = cur
            self.tail = cur
        self.list_length += 1

    def prepend(self, item):
        """Insert the given item at the head of this linked list.
        time:   Ω(1) = O(1) -> Θ(1) all we have to do is set head to new node."""
        cur = Node(item)
        if self.head is None:
            self.head = self.tail = cur
        else:
            cur.next = self.head
            self.head = cur
        self.list_length += 1

    def find(self, quality):
        """Return an item from this linked list satisfying the given quality.
        time:   Ω(1) = O(1) -> Θ(1) if quality matches head.
                Ω(n) = O(n) -> Θ(n) if quality matches tail."""
        cur = self.head
        while cur is not None and not quality(cur.data):
            cur = cur.next
        if cur is not None:
            return cur.data

    def delete(self, item):
        """Delete the given item from this linked list, or raise ValueError.
        time:   Ω(1) = O(1) -> Θ(1) if item is head.
                Ω(n) = O(n) -> Θ(n) if item is tail."""
        prev = None
        cur = self.head

        while cur is not None and cur.data != item:
            prev = cur
            cur = cur.next

        if cur is None:
            # item wasn't found
            raise ValueError(f"{item} wasn't found :(")

        if cur is self.head:
            if self.head is self.tail:
                self.head = self.tail = None
            else:
                self.head = self.head.next
        elif cur is self.tail:
            self.tail = prev
            self.tail.next = None
        else:
            prev.next = cur.next
        self.list_length -= 1

    def replace(self, quality, new_item=None):
        """Takes a quality that must be satisfied in order to replace with
        new_item. In the case that quality isn't found, append new_item to the
        end. In the case that new_item is None, delete item matching given
        quality.
        time:   Ω(1) = O(1) -> Θ(1) if quality matches head.
                Ω(n) = O(n) -> Θ(n) if quality matches tail."""
        prev = None
        cur = self.head

        while cur is not None and not quality(cur.data):
            prev = cur
            cur = cur.next

        if cur is not None:
            # old_item must've been found
            if new_item is None:
                # we're deleting cur
                if cur is self.head:
                    if self.head is self.tail:
                        self.head = self.tail = None
                    else:
                        self.head = self.head.next
                elif cur is self.tail:
                    # we've already handled the special case where length is 1
                    prev.next = None
                    self.tail = prev
                else:
                    prev.next = cur.next
                self.list_length -= 1
            else:
                cur.data = new_item
        elif new_item is not None:
            self.append(new_item)
        else:
            raise KeyError(f"item matching quality wasn't found!")


def test_linked_list():
    ll = LinkedList()
    print('list: {}'.format(ll))

    print('\nTesting append:')
    for item in ['A', 'B', 'C']:
        print('append({!r})'.format(item))
        ll.append(item)
        print('list: {}'.format(ll))

    print('head: {}'.format(ll.head))
    print('tail: {}'.format(ll.tail))
    print('length: {}'.format(ll.length()))

    # Enable this after implementing delete method
    delete_implemented = False
    if delete_implemented:
        print('\nTesting delete:')
        for item in ['B', 'C', 'A']:
            print('delete({!r})'.format(item))
            ll.delete(item)
            print('list: {}'.format(ll))

        print('head: {}'.format(ll.head))
        print('tail: {}'.format(ll.tail))
        print('length: {}'.format(ll.length()))


if __name__ == '__main__':
    test_linked_list()
