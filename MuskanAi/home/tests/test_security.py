## Security test
from django.test import TestCase
from django.contrib.auth.models import User
from django.urls import reverse


class SecurityTests(TestCase):
    """Contains the security test cases for the app"""

    def csrfTest(self):
        """Test to check if the csrf token is present in the response"""
        response = self.client.get('/login')
        response2 = self.client.get('/signup')
        self.assertContains(response, 'csrfmiddlewaretoken')
        self.assertContains(response2, 'csrfmiddlewaretoken')

        