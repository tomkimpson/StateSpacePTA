Usage
=====

.. _installation:

Installation
------------

To use Lumache, first install it using pip:

.. code-block:: console

   (.venv) $ pip install lumache




Creating recipes
----------------


.. py:exception:: lumache.InvalidKindError

   Raised if the kind is invalid.



To retrieve a list of random ingredients,
you can use the ``lumache.get_random_ingredients()`` function:


you can use the ``lumache.get_random_ingredients()`` function:

.. autofunction:: lumache.get_random_ingredients
    

The ``kind`` parameter should be either ``"meat"``, ``"fish"``,
or ``"veggies"``. Otherwise, :py:func:`lumache.get_random_ingredients`
will raise an exception.